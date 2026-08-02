import importlib
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_THIS_DIR = Path(__file__).resolve().parent
_SLACLIP_DIR = _THIS_DIR.parent
_REPO_ROOT = _SLACLIP_DIR.parent
_PATCHES_DIR = _SLACLIP_DIR / "patches"

sys.path.insert(0, str(_PATCHES_DIR))
if str(_SLACLIP_DIR) not in sys.path:
    sys.path.insert(1, str(_SLACLIP_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(2, str(_REPO_ROOT))

for _module_name in list(sys.modules):
    if _module_name == "opacus" or _module_name.startswith("opacus."):
        sys.modules.pop(_module_name, None)

from opacus.optimizers.slaclipoptimizer import SlaClipOptimizer, SlaClipQOptimizer
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from slaclip.args import (
    PAPER_PRACTICAL_K_SIGMA1,
    build_parser,
    paper_recommended_k,
)
from slaclip.train_loop import train_one_epoch

slaclipoptimizer_module = importlib.import_module(
    "opacus.optimizers.slaclipoptimizer"
)


def _make_slaclip_optimizer(
    *,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 2.0,
    expected_batch_size: int = 1,
    num_slots: int = 2,
    eta: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
    c_min: float = 0.1,
    c_max: float = 20.0,
    strict_paper_check: bool = True,
):
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dp = SlaClipOptimizer(
        opt,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
        num_slots=num_slots,
        eta=eta,
        beta=beta,
        gamma=gamma,
        c_min=c_min,
        c_max=c_max,
        strict_paper_check=strict_paper_check,
    )
    return model, dp


def test_parser_leaves_k_unset_for_auto_selection():
    parser = build_parser()
    args = parser.parse_args(["--method", "slaclip", "--dataset", "mnist"])
    assert args.K is None


@pytest.mark.parametrize(
    ("expected_batch_size", "sigma"),
    [(128, 1.0), (512, 1.0), (1024, 1.317), (2048, 2.34)],
)
def test_paper_recommended_k_uses_equation_14(expected_batch_size, sigma):
    z_0995 = 2.5758293035489004
    expected = max(
        1,
        math.floor(
            (expected_batch_size / (2.0 * z_0995 * sigma)) ** (2.0 / 3.0)
        ),
    )
    assert paper_recommended_k(expected_batch_size, sigma) == expected


def test_appendix_d_sigma1_practical_k_values_are_preserved():
    assert PAPER_PRACTICAL_K_SIGMA1 == {
        128: 8,
        256: 10,
        512: 20,
        1024: 30,
        2048: 50,
    }


def test_slaclip_update_equation():
    _model, dp = _make_slaclip_optimizer()
    s_hat = torch.tensor([0.2, 0.4])
    C_t = 2.0
    r_t = max(0.0, min(1.0, 0.4 / C_t))
    gamma_t = max(0.0, min(1.0, 1.0 - (1.0 - r_t) / 2.0))
    expected = C_t * math.exp(0.5 * (gamma_t - 0.2))
    actual = dp._update_threshold(C_t, s_hat)
    assert abs(actual - expected) < 1e-6


@pytest.mark.parametrize(
    ("s_last", "expected_gamma"),
    [
        (-10.0, 0.5),  # Eq. (28) clips negative noisy feedback before Eq. (29)
        (0.4, 0.6),
        (2.0, 1.0),
        (20.0, 1.0),
    ],
)
def test_slaclip_appendix_c_clips_last_slot_before_gamma(s_last, expected_gamma):
    _model, dp = _make_slaclip_optimizer(eta=1.0)
    C_t = 2.0
    s_first = 0.3
    actual = dp._update_threshold(C_t, torch.tensor([s_first, s_last]))
    expected = C_t * math.exp(expected_gamma - s_first)
    assert actual == pytest.approx(expected)


def test_strict_paper_mode_requires_fixed_half_feedback():
    with pytest.raises(ValueError, match="requires beta=0.5"):
        _make_slaclip_optimizer(beta=0.4, strict_paper_check=True)


def test_strict_threshold_guardrails_reject_invalid_initial_value():
    with pytest.raises(ValueError, match="max_grad_norm/C0 inside"):
        _make_slaclip_optimizer(max_grad_norm=100.0)


def test_threshold_update_guardrails_avoid_overflow():
    _model, dp = _make_slaclip_optimizer(max_grad_norm=2.0)
    assert dp._update_threshold(2.0, torch.tensor([-1.0e30, 2.0])) == 20.0
    assert dp._update_threshold(2.0, torch.tensor([1.0e30, 2.0])) == 0.1


def test_slaclip_q_respects_cli_bounds():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dp = SlaClipQOptimizer(
        opt,
        noise_multiplier=1.0,
        max_grad_norm=2.0,
        expected_batch_size=1,
        num_slots=2,
        eta=10.0,
        gamma=0.99,
        c_min=0.5,
        c_max=3.0,
        strict_paper_check=False,
    )
    assert dp._update_threshold(2.0, torch.tensor([0.0, 0.0])) == 3.0


def test_slack_vector_preserves_extended_norm_bound():
    _model, dp = _make_slaclip_optimizer(max_grad_norm=2.0, num_slots=4)
    C_t = 2.0
    norms = torch.tensor([0.0, 0.5, 1.25, 2.0, 3.0], dtype=torch.float32)
    slack_amount = torch.clamp(C_t - norms, min=0.0)
    lambda_t = C_t / math.sqrt(dp.K)
    slack_vector = dp._build_slack_vector(slack_amount * math.sqrt(dp.K), lambda_t)
    clipped_norms = torch.clamp(norms, max=C_t)
    extended_norms = torch.sqrt(clipped_norms.square() + slack_vector.square().sum(dim=1))
    assert torch.all(extended_norms <= (C_t + 1e-6))


def test_slack_indicator_uses_expected_batch_size_under_mean_reduction():
    model, dp = _make_slaclip_optimizer(
        noise_multiplier=0.0,
        max_grad_norm=2.0,
        expected_batch_size=8,
        num_slots=2,
    )
    param = next(model.parameters())
    param.grad_sample = torch.zeros((4,) + tuple(param.shape), dtype=param.dtype)
    param.summed_grad = torch.zeros_like(param)

    dp._slack_sum = torch.tensor([16.0, 8.0], dtype=torch.float32)
    dp._lambda_t = 2.0

    dp.add_noise()

    expected = torch.tensor([1.0, 0.5], dtype=torch.float32)
    assert torch.allclose(dp._slack_indicator, expected)


def test_gradient_and_slack_use_exactly_one_joint_noise_draw(monkeypatch):
    calls = []

    def deterministic_noise(*, std, reference, generator, secure_mode):
        calls.append((std, reference.numel()))
        return torch.arange(
            reference.numel(), device=reference.device, dtype=reference.dtype
        )

    monkeypatch.setattr(
        slaclipoptimizer_module, "_generate_noise", deterministic_noise
    )

    model, dp = _make_slaclip_optimizer(
        noise_multiplier=1.0,
        max_grad_norm=2.0,
        expected_batch_size=1,
        num_slots=2,
    )
    param = next(model.parameters())
    param.grad_sample = torch.zeros((1,) + tuple(param.shape), dtype=param.dtype)
    param.summed_grad = torch.tensor([[3.0]], dtype=param.dtype)
    dp._slack_sum = torch.zeros(2, dtype=torch.float32)
    dp._lambda_t = 1.0

    dp.add_noise()

    # One vector of length d+K is sampled; its prefix noises the gradient and
    # its suffix noises slack. There is no separately sampled slack query.
    assert calls == [(2.0, 3)]
    assert torch.equal(param.grad, torch.tensor([[3.0]]))
    assert torch.equal(dp._slack_indicator, torch.tensor([1.0, 2.0]))
    assert not hasattr(dp, "_compute_slack_indicator")


def test_empty_poisson_batch_still_releases_noise_and_updates_accounted_step():
    model, dp = _make_slaclip_optimizer(
        noise_multiplier=0.0,
        max_grad_norm=2.0,
        expected_batch_size=8,
        num_slots=2,
    )
    param = next(model.parameters())
    param.grad_sample = torch.empty((0,) + tuple(param.shape), dtype=param.dtype)

    assert dp.pre_step() is True
    assert torch.equal(param.grad, torch.zeros_like(param))
    assert torch.equal(dp._slack_indicator, torch.zeros(2))
    assert dp._sample_count == 0
    assert dp.current_clip == pytest.approx(2.0 * math.exp(0.5 * 0.5))


def test_slack_sum_accumulates_across_clip_calls():
    model, dp = _make_slaclip_optimizer(
        noise_multiplier=0.0,
        max_grad_norm=2.0,
        expected_batch_size=4,
        num_slots=4,
    )
    param = next(model.parameters())

    param.grad_sample = torch.tensor([[[0.5]], [[0.5]]], dtype=param.dtype)
    dp.clip_and_accumulate()

    param.grad_sample = torch.tensor([[[1.0]], [[1.0]]], dtype=param.dtype)
    dp.clip_and_accumulate()

    expected = torch.tensor([4.0, 4.0, 2.0, 0.0], dtype=torch.float32)
    assert torch.equal(dp._slack_sum, expected)
    assert dp._sample_count == 4


def test_virtual_batches_accumulate_slack_until_one_logical_release():
    model, dp = _make_slaclip_optimizer(
        noise_multiplier=0.0,
        max_grad_norm=2.0,
        expected_batch_size=4,
        num_slots=4,
    )
    param = next(model.parameters())

    param.grad_sample = torch.tensor([[[0.5]], [[0.5]]], dtype=param.dtype)
    dp.signal_skip_step(do_skip=True)
    assert dp.step() is None
    dp.zero_grad()

    assert torch.equal(dp._slack_sum, torch.tensor([2.0, 2.0, 2.0, 0.0]))
    assert torch.equal(param.summed_grad, torch.tensor([[1.0]]))

    param.grad_sample = torch.tensor([[[1.0]], [[1.0]]], dtype=param.dtype)
    dp.signal_skip_step(do_skip=False)
    dp.step()

    assert torch.equal(dp._slack_sum, torch.tensor([4.0, 4.0, 2.0, 0.0]))
    assert torch.equal(dp._slack_indicator, torch.tensor([1.0, 1.0, 0.5, 0.0]))
    assert dp._sample_count == 4

    dp.zero_grad()
    assert dp._slack_sum is None
    assert param.summed_grad is None


@pytest.mark.parametrize("optimizer_cls", [SlaClipOptimizer, SlaClipQOptimizer])
def test_controller_state_dict_round_trip_at_logical_step_boundary(optimizer_cls):
    def make_optimizer():
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        dp_optimizer = optimizer_cls(
            optimizer,
            noise_multiplier=0.0,
            max_grad_norm=2.0,
            expected_batch_size=1,
            num_slots=2,
            eta=0.5,
            beta=0.5,
            gamma=0.5,
            c_min=0.1,
            c_max=20.0,
            strict_paper_check=True,
        )
        return model, dp_optimizer

    model, dp = make_optimizer()
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.25]]], dtype=param.dtype)
    dp.step()
    checkpoint_clip = dp.current_clip
    checkpoint = dp.state_dict()

    _restored_model, restored = make_optimizer()
    restored.load_state_dict(checkpoint)

    assert restored.current_clip == checkpoint_clip
    assert restored.max_grad_norm == checkpoint_clip
    assert restored._slack_sum is None
    assert restored._slack_indicator is None


def test_controller_checkpoint_rejects_inflight_virtual_batch():
    model, dp = _make_slaclip_optimizer(noise_multiplier=0.0)
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.25]]], dtype=param.dtype)
    dp.signal_skip_step(do_skip=True)
    dp.step()

    with pytest.raises(RuntimeError, match="unfinished private query"):
        dp.state_dict()


def test_controller_checkpoint_rejects_backward_before_dp_release():
    model, dp = _make_slaclip_optimizer(noise_multiplier=0.0)
    param = next(model.parameters())
    param.grad_sample = torch.tensor([[[0.25]]], dtype=param.dtype)

    with pytest.raises(RuntimeError, match="unfinished private query"):
        dp.state_dict()


def test_controller_checkpoint_rejects_configuration_mismatch():
    _model, dp = _make_slaclip_optimizer(num_slots=2)
    checkpoint = dp.state_dict()
    _other_model, other = _make_slaclip_optimizer(num_slots=3)

    with pytest.raises(ValueError, match="configuration mismatch for K"):
        other.load_state_dict(checkpoint)


def test_real_batch_memory_manager_has_one_release_and_controller_update_per_logical_batch():
    torch.manual_seed(2026)
    model = torch.nn.Linear(2, 2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 2), torch.randint(0, 2, (16,))),
        batch_size=8,
        shuffle=False,
    )
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=base_optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        clipping="slaclip",
        num_slots=2,
        eta=0.5,
        beta=0.5,
        gamma=0.5,
        c_min=0.1,
        c_max=20.0,
        strict_paper_check=True,
        poisson_sampling=True,
        grad_sample_mode="hooks",
    )

    physical_steps = 0
    controller_updates = 0
    original_step = optimizer.step
    original_update = optimizer._update_threshold

    def counted_step(*args, **kwargs):
        nonlocal physical_steps
        physical_steps += 1
        return original_step(*args, **kwargs)

    def counted_update(*args, **kwargs):
        nonlocal controller_updates
        controller_updates += 1
        return original_update(*args, **kwargs)

    optimizer.step = counted_step
    optimizer._update_threshold = counted_update

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
    assert stopped is False
    assert logical_steps == len(private_loader) == 2
    assert accounted_steps == logical_steps
    assert controller_updates == logical_steps
    assert physical_steps > logical_steps
