import importlib
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_THIS_DIR = Path(__file__).resolve().parent
_SLACLIP_DIR = _THIS_DIR.parent
_PATCHES_DIR = _SLACLIP_DIR / "patches"

sys.path.insert(0, str(_PATCHES_DIR))

# The audit environment also has upstream Opacus installed. Ensure this test
# exercises the overlay implementations from this checkout.
for _module_name in list(sys.modules):
    if _module_name == "opacus" or _module_name.startswith("opacus."):
        sys.modules.pop(_module_name, None)

from opacus.optimizers.DCSGDEOptimizer import DCSGDEOptimizer
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.autoclipoptimizer import AutoClipDPOptimizer
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from slaclip.train_loop import train_one_epoch

adaclipoptimizer_module = importlib.import_module("opacus.optimizers.adaclipoptimizer")
dcsgdeoptimizer_module = importlib.import_module("opacus.optimizers.DCSGDEOptimizer")


def _linear_optimizer():
    model = torch.nn.Linear(2, 1, bias=False)
    return model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


def _make_adaclip(**overrides):
    model, optimizer = _linear_optimizer()
    kwargs = {
        "noise_multiplier": 1.0,
        "target_unclipped_quantile": 0.5,
        "clipbound_learning_rate": 0.2,
        "max_clipbound": 20.0,
        "min_clipbound": 0.1,
        "unclipped_num_std": 1.0,
        "max_grad_norm": 2.0,
        "expected_batch_size": 2,
    }
    kwargs.update(overrides)
    return model, AdaClipDPOptimizer(optimizer, **kwargs)


def _make_dcsgde(**overrides):
    model, optimizer = _linear_optimizer()
    kwargs = {
        "noise_multiplier": 1.0,
        "histogram_std": 6.0,
        "max_grad_norm": 2.0,
        "expected_batch_size": 2,
        "batchsize_train": 256,
        "dimension": 2,
        "percentile": 0.3,
        "stride": 1.0,
        "bin_cnt": 20,
        "c_min": 0.1,
        "c_max": 20.0,
    }
    kwargs.update(overrides)
    return model, DCSGDEOptimizer(optimizer, **kwargs)


def test_adaclip_noise_composition_and_accountant_visible_sigma():
    sigma_total = 1.0
    sigma_b = 1.0
    model, dp = _make_adaclip(
        noise_multiplier=sigma_total,
        unclipped_num_std=sigma_b,
    )

    expected_sigma_grad = (sigma_total**-2 - (2.0 * sigma_b) ** -2) ** -0.5
    assert dp._grad_noise_multiplier == pytest.approx(expected_sigma_grad)

    param = next(model.parameters())
    param.grad_sample = torch.zeros((1,) + tuple(param.shape), dtype=param.dtype)
    param.summed_grad = torch.zeros_like(param)
    observed = []
    dp.attach_step_hook(lambda optimizer: observed.append(optimizer.noise_multiplier))
    dp.add_noise()
    dp.step_hook(dp)

    assert observed == [sigma_total]
    assert dp.noise_multiplier == sigma_total


def test_adaclip_projects_noisy_fraction_to_unit_interval():
    _model, dp = _make_adaclip(max_grad_norm=2.0)
    dp.noisy_unclipped_fraction = 0.0

    dp.update_max_grad_norm()

    # AdaClip projects the *noisy statistic* to [0, 1]. The public target
    # quantile is separately restricted to [0.01, 0.99].
    expected = 2.0 * math.exp(-0.2 * (0.0 - 0.5))
    assert dp.max_grad_norm == pytest.approx(expected)


def test_adaclip_state_dict_roundtrip_restores_controller():
    model, source = _make_adaclip()
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.2, 0.1]]], dtype=param.dtype)
    source.step()
    completed_step_c = source.max_grad_norm

    state = source.state_dict()
    assert set(state["_slaclip_adaclip_state"]) == {"version", "max_grad_norm"}

    _model, restored = _make_adaclip()
    restored.load_state_dict(state)

    assert restored.max_grad_norm == completed_step_c
    assert restored.sample_size == 0
    assert restored.centered_unclipped_sum == 0.0
    assert restored.noisy_unclipped_fraction is None


def test_adaclip_rejects_checkpoint_with_unreleased_private_count():
    model, dp = _make_adaclip()
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.2, 0.1]]], dtype=param.dtype)

    with pytest.raises(RuntimeError, match="incomplete logical batch"):
        dp.state_dict()

    dp.clip_and_accumulate()
    with pytest.raises(RuntimeError, match="unclipped-count query"):
        dp.state_dict()


def test_adaclip_count_query_accumulates_across_physical_batches():
    model, dp = _make_adaclip(noise_multiplier=0.0)
    param = next(model.parameters())
    first_physical_batch = torch.tensor([[[0.1, 0.1]]], dtype=param.dtype)
    second_physical_batch = torch.tensor([[[0.2, 0.2]]], dtype=param.dtype)

    param.grad_sample = first_physical_batch
    dp.clip_and_accumulate()
    dp._is_last_step_skipped = True
    dp.zero_grad()

    assert dp.sample_size == 1
    assert dp.centered_unclipped_sum == 0.5
    with pytest.raises(RuntimeError, match="incomplete logical batch"):
        dp.state_dict()

    param.grad_sample = second_physical_batch
    dp.clip_and_accumulate()

    assert dp.sample_size == 2
    assert dp.centered_unclipped_sum == 1.0

    dp._is_last_step_skipped = False
    dp.zero_grad()
    assert dp.sample_size == 0
    assert dp.centered_unclipped_sum == 0.0


def test_adaclip_poisson_count_uses_fixed_expected_denominator(monkeypatch):
    monkeypatch.setattr(
        adaclipoptimizer_module,
        "_generate_noise",
        lambda **kwargs: torch.zeros_like(kwargs["reference"]),
    )

    model, dp = _make_adaclip(
        noise_multiplier=0.0,
        expected_batch_size=8,
    )
    param = next(model.parameters())
    # Two realized records are both unclipped, so the centered query is 1.
    # The public denominator remains expected B=8 rather than realized m=2.
    param.grad_sample = torch.tensor([[[0.1, 0.1]], [[0.2, 0.2]]], dtype=param.dtype)
    dp.clip_and_accumulate()
    dp.add_noise()

    assert dp.noisy_unclipped_fraction == pytest.approx(0.5 + 1.0 / 8.0)
    assert dp.sample_size == 0
    assert dp.centered_unclipped_sum == 0.0


@pytest.mark.parametrize("quantile", [0.0, 1.0])
def test_adaclip_rejects_extreme_public_quantiles(quantile):
    with pytest.raises(ValueError, match=r"\[0\.01, 0\.99\]"):
        _make_adaclip(target_unclipped_quantile=quantile)


def test_autoclip_auto_s_formula_has_no_abadi_clamp():
    model, optimizer = _linear_optimizer()
    dp = AutoClipDPOptimizer(
        optimizer=optimizer,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=1,
        gamma=0.01,
        mode="auto_s",
    )
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.3, 0.4]]], dtype=param.dtype)

    dp.clip_and_accumulate()

    expected = torch.tensor([[0.3, 0.4]], dtype=param.dtype) / 0.51
    assert torch.allclose(param.summed_grad, expected)
    assert param.summed_grad.norm() > torch.tensor(0.5)


@pytest.mark.parametrize("radius", [-1.0, 0.0, 0.05, 20.1])
def test_autoclip_requires_positive_in_range_radius(radius):
    _model, optimizer = _linear_optimizer()
    with pytest.raises(ValueError, match="R must"):
        AutoClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=radius,
            expected_batch_size=1,
        )


def test_empty_poisson_batch_releases_count_noise_and_updates_adaclip(
    monkeypatch,
):
    count_noise_calls = []

    def fixed_count_noise(**kwargs):
        count_noise_calls.append(float(kwargs["std"]))
        return torch.ones_like(kwargs["reference"])

    monkeypatch.setattr(
        adaclipoptimizer_module,
        "_generate_noise",
        fixed_count_noise,
    )
    model, dp = _make_adaclip(
        noise_multiplier=1.0,
        generator=torch.Generator().manual_seed(1001),
    )
    param = next(model.parameters())
    param.grad_sample = torch.empty((0,) + tuple(param.shape), dtype=param.dtype)
    initial_c = dp.max_grad_norm
    observed = []
    dp.attach_step_hook(lambda optimizer: observed.append(optimizer.noise_multiplier))

    assert dp.pre_step() is True

    assert torch.isfinite(param.grad).all()
    assert torch.count_nonzero(param.grad) > 0
    assert observed == [1.0]
    assert count_noise_calls == [1.0]
    expected_fraction = 1.0  # clip(0.5 + 1 / expected_B=2, 0, 1)
    expected_c = initial_c * math.exp(-0.2 * (expected_fraction - 0.5))
    assert dp.max_grad_norm == pytest.approx(expected_c)
    assert dp.sample_size == 0
    assert dp.centered_unclipped_sum == 0.0
    assert dp.noisy_unclipped_fraction is None
    assert dp._controller_update_pending is False


def test_empty_poisson_batch_releases_gradient_for_autoclip():
    model, optimizer = _linear_optimizer()
    dp = AutoClipDPOptimizer(
        optimizer=optimizer,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
        generator=torch.Generator().manual_seed(1002),
    )
    param = next(model.parameters())
    param.grad_sample = torch.empty((0,) + tuple(param.shape), dtype=param.dtype)
    observed = []
    dp.attach_step_hook(lambda optimizer: observed.append(optimizer.noise_multiplier))

    assert dp.pre_step() is True

    assert torch.isfinite(param.grad).all()
    assert torch.count_nonzero(param.grad) > 0
    assert observed == [1.0]


def test_dcsgde_noise_composition_and_accountant_visible_sigma():
    sigma_total = 1.0
    sigma_hist = 6.0
    model, dp = _make_dcsgde(
        noise_multiplier=sigma_total,
        histogram_std=sigma_hist,
    )

    expected_sigma_grad = (sigma_total**-2 - sigma_hist**-2) ** -0.5
    assert dp._grad_noise_multiplier == pytest.approx(expected_sigma_grad)

    param = next(model.parameters())
    param.summed_grad = torch.zeros_like(param)
    observed = []
    dp.attach_step_hook(lambda optimizer: observed.append(optimizer.noise_multiplier))
    dp.add_noise()
    dp.step_hook(dp)

    assert observed == [sigma_total]
    assert dp.noise_multiplier == sigma_total


def test_dcsgde_boundary_search_stops_when_hard_bound_blocks_expansion():
    _model, dp = _make_dcsgde(
        max_grad_norm=20.0,
        stride=100.0,
        dimension=1,
        batchsize_train=10**9,
        max_boundary_search_iterations=4,
    )
    dp.norm_stack = [10_000.0]
    dp._scalar_hist_noise = lambda device: 0.0

    dp.before_clip()

    assert dp.max_grad_norm == 20.0
    assert dp._last_boundary_search_iterations == 1
    assert dp.norm_stack == []


def test_dcsgde_boundary_search_obeys_explicit_iteration_cap():
    _model, dp = _make_dcsgde(
        max_grad_norm=1.0,
        stride=100.0,
        dimension=1,
        batchsize_train=10**9,
        max_boundary_search_iterations=2,
    )
    dp.norm_stack = [10_000.0]
    dp._scalar_hist_noise = lambda device: 0.0

    dp.before_clip()

    assert dp._last_boundary_search_iterations == 2
    assert 0.1 <= dp.max_grad_norm <= 20.0


def test_dcsgde_state_dict_roundtrip_restores_controller():
    model, source = _make_dcsgde()
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.2, 0.1]]], dtype=param.dtype)
    source.step()
    completed_state = (
        source.max_grad_norm,
        source.timer,
        source.stride,
        source._last_boundary_search_iterations,
    )

    state = source.state_dict()
    assert set(state["_slaclip_dcsgde_state"]) == {
        "version",
        "max_grad_norm",
        "timer",
        "stride",
        "last_boundary_search_iterations",
    }

    _model, restored = _make_dcsgde()
    restored.load_state_dict(state)

    assert (
        restored.max_grad_norm,
        restored.timer,
        restored.stride,
        restored._last_boundary_search_iterations,
    ) == completed_state
    assert restored.sample_size == 0
    assert restored.unclipped_num == 0
    assert restored.norm_stack == []
    assert restored._histogram_query_pending is False


def test_dcsgde_rejects_checkpoint_with_raw_per_sample_norms():
    model, dp = _make_dcsgde()
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.2, 0.1]]], dtype=param.dtype)

    with pytest.raises(RuntimeError, match="incomplete logical batch"):
        dp.state_dict()

    dp.clip_and_accumulate()
    assert dp.norm_stack
    with pytest.raises(RuntimeError, match="raw per-sample gradient norms"):
        dp.state_dict()


def test_dcsgde_rejects_checkpoint_after_empty_clip_before_noise_release():
    model, dp = _make_dcsgde()
    param = next(model.parameters())
    param.grad_sample = torch.empty((0,) + tuple(param.shape), dtype=param.dtype)
    dp.clip_and_accumulate()

    with pytest.raises(RuntimeError, match="incomplete logical batch"):
        dp.state_dict()


def test_empty_poisson_batch_releases_gradient_and_histogram_noise(monkeypatch):
    histogram_noise_calls = []

    def positive_histogram_noise(*, device):
        del device
        histogram_noise_calls.append(1)
        return 1.0

    model, dp = _make_dcsgde(
        noise_multiplier=1.0,
        generator=torch.Generator().manual_seed(1003),
    )
    monkeypatch.setattr(dp, "_scalar_hist_noise", positive_histogram_noise)
    param = next(model.parameters())
    param.grad_sample = torch.empty((0,) + tuple(param.shape), dtype=param.dtype)
    observed = []
    dp.attach_step_hook(lambda optimizer: observed.append(optimizer.noise_multiplier))

    assert dp.pre_step() is True

    assert torch.isfinite(param.grad).all()
    assert torch.count_nonzero(param.grad) > 0
    assert observed == [1.0]
    assert len(histogram_noise_calls) == dp.bin_cnt
    assert dp._last_boundary_search_iterations >= 1
    assert dp.norm_stack == []
    assert dp._histogram_query_pending is False


@pytest.mark.parametrize("clipping", ["adaptive", "dc-sgd-e"])
def test_real_batch_memory_manager_releases_baseline_controller_once_per_logical_step(
    monkeypatch,
    clipping,
):
    torch.manual_seed(2026)
    model = torch.nn.Linear(2, 2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 2), torch.randint(0, 2, (16,))),
        batch_size=8,
        shuffle=False,
    )
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    method_kwargs = {}
    if clipping == "adaptive":
        method_kwargs = {
            "target_unclipped_quantile": 0.5,
            "clipbound_learning_rate": 0.2,
            "max_clipbound": 20.0,
            "min_clipbound": 0.1,
            "unclipped_num_std": 1.0,
        }
    else:
        method_kwargs = {
            "histogram_std": 6.0,
            "batchsize_train": 8,
            "dimension": 6,
            "percentile": 0.3,
            "stride": 1.0,
            "bin_cnt": 20,
            "c_min": 0.1,
            "c_max": 20.0,
        }

    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=base_optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        clipping=clipping,
        poisson_sampling=True,
        grad_sample_mode="hooks",
        **method_kwargs,
    )

    release_calls = 0
    if clipping == "adaptive":
        original_release = adaclipoptimizer_module._generate_noise

        def counted_release(**kwargs):
            nonlocal release_calls
            release_calls += 1
            return original_release(**kwargs)

        monkeypatch.setattr(
            adaclipoptimizer_module,
            "_generate_noise",
            counted_release,
        )
    else:
        original_release = optimizer._scalar_hist_noise

        def counted_release(device):
            nonlocal release_calls
            release_calls += 1
            return original_release(device)

        monkeypatch.setattr(optimizer, "_scalar_hist_noise", counted_release)

    physical_steps = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal physical_steps
        physical_steps += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step

    with BatchMemoryManager(
        data_loader=private_loader,
        max_physical_batch_size=2,
        optimizer=optimizer,
    ) as memory_safe_loader:
        _, _, stopped, logical_steps = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=memory_safe_loader,
            device=torch.device("cpu"),
            criterion=torch.nn.CrossEntropyLoss(),
            epoch=1,
            privacy_engine=privacy_engine,
            delta=1e-5,
            expose_training_metrics=False,
        )

    accounted_steps = sum(int(entry[2]) for entry in privacy_engine.accountant.history)
    expected_releases = logical_steps if clipping == "adaptive" else 20 * logical_steps
    assert stopped is False
    assert logical_steps == len(private_loader) == 2
    assert accounted_steps == logical_steps
    assert release_calls == expected_releases
    assert physical_steps > logical_steps
