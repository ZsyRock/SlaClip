import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


_SLACLIP_DIR = Path(__file__).resolve().parents[1]
_PATCHES_DIR = _SLACLIP_DIR / "patches"
_OPACUS_ROOT = _SLACLIP_DIR.parent
sys.path.insert(0, str(_PATCHES_DIR))
sys.path.insert(1, str(_SLACLIP_DIR))
sys.path.insert(2, str(_OPACUS_ROOT))

for _module_name in list(sys.modules):
    if _module_name == "opacus" or _module_name.startswith("opacus."):
        sys.modules.pop(_module_name, None)

from opacus import PrivacyEngine


def _model_and_optimizer():
    model = torch.nn.Linear(2, 2)
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


@pytest.mark.parametrize("privacy_mode", ["strict", "research"])
def test_paper_method_rejects_ghost_before_mutating_module(privacy_mode):
    model, optimizer = _model_and_optimizer()
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    engine = PrivacyEngine(
        accountant="rdp", secure_mode=False, privacy_mode=privacy_mode
    )
    hooks_before = len(model._forward_hooks)

    with pytest.raises(ValueError, match="requires grad_sample_mode='hooks'"):
        engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            clipping="slaclip",
            grad_sample_mode="ghost",
            num_slots=2,
        )

    assert len(model._forward_hooks) == hooks_before
    assert all(not hasattr(parameter, "grad_sample") for parameter in model.parameters())


def test_strict_mode_rejects_non_poisson_sampling_before_wrapping():
    model, optimizer = _model_and_optimizer()
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    engine = PrivacyEngine(accountant="rdp", secure_mode=False)

    with pytest.raises(ValueError, match="require Poisson sampling"):
        engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )


@pytest.mark.parametrize(
    ("clipping", "kwargs"),
    [
        ("slaclip", {"num_slots": 2, "nmu_slots": 2}),
        (
            "adaptive",
            {
                "target_unclipped_quantile": 0.5,
                "clipbound_learning_rate": 0.2,
                "max_clipbound": 20.0,
                "min_clipbound": 0.1,
                "unclipped_num_std": 1.0,
                "target_unclipped_quantilee": 0.5,
            },
        ),
        (
            "dc-sgd-e",
            {
                "batchsize_train": 2,
                "dimension": 6,
                "histogram_std": 5.0,
                "bin_cnut": 20,
            },
        ),
    ],
)
def test_strict_custom_optimizers_reject_misspelled_options(clipping, kwargs):
    _model, optimizer = _model_and_optimizer()
    engine = PrivacyEngine(accountant="rdp", secure_mode=False)

    with pytest.raises(TypeError, match="unsupported keyword arguments"):
        engine._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=2,
            clipping=clipping,
            grad_sample_mode="hooks",
            **kwargs,
        )


def test_make_private_with_epsilon_separates_calibration_and_optimizer_kwargs(
    monkeypatch,
):
    calibration_call = {}

    def fake_get_noise_multiplier(**kwargs):
        calibration_call.update(kwargs)
        return 1.25

    # Several overlay tests deliberately reload ``opacus`` during collection.
    # Patch the exact globals dictionary used by this class object rather than
    # whichever later module object currently occupies ``sys.modules``.
    monkeypatch.setitem(
        PrivacyEngine.make_private_with_epsilon.__globals__,
        "get_noise_multiplier",
        fake_get_noise_multiplier,
    )

    model, optimizer = _model_and_optimizer()
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )
    engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    make_private_call = {}

    def fake_make_private(**kwargs):
        make_private_call.update(kwargs)
        return "wrapped"

    monkeypatch.setattr(engine, "make_private", fake_make_private)

    result = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=3.0,
        target_delta=1e-5,
        epochs=2,
        max_grad_norm=1.0,
        clipping="slaclip",
        num_slots=7,
        eta=0.2,
        beta=0.5,
        gamma=0.5,
        c_min=0.1,
        c_max=20.0,
        strict_paper_check=True,
        epsilon_tolerance=0.005,
    )

    assert result == "wrapped"
    assert calibration_call["epsilon_tolerance"] == 0.005
    assert calibration_call["steps"] == len(loader) * 2 == 4
    assert "epochs" not in calibration_call
    assert "num_slots" not in calibration_call
    assert make_private_call["num_slots"] == 7
    assert make_private_call["noise_multiplier"] == 1.25


def test_poisson_conversion_preserves_211_logical_steps_and_sample_rate():
    model, optimizer = _model_and_optimizer()
    loader = DataLoader(
        TensorDataset(torch.zeros(422, 2), torch.zeros(422, dtype=torch.long)),
        batch_size=2,
    )
    assert len(loader) == 211
    engine = PrivacyEngine(accountant="rdp", secure_mode=False)

    _model, private_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        clipping="flat",
        poisson_sampling=True,
    )

    assert len(private_loader) == 211
    assert private_loader.batch_sampler.steps == 211
    assert private_loader.sample_rate == 1.0 / 211.0
    assert engine.sample_rate == 1.0 / 211.0
    assert private_optimizer.expected_batch_size == 2
