import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

data_stub = ModuleType("slaclip.data")
data_stub.make_dataloaders = object()
sys.modules["slaclip.data"] = data_stub

import run_exp


def test_runner_calibrates_with_exact_integer_release_count(monkeypatch):
    captured = {}

    def fake_get_noise_multiplier(**kwargs):
        captured.update(kwargs)
        return 1.25

    monkeypatch.setattr(run_exp, "get_noise_multiplier", fake_get_noise_multiplier)
    loader = DataLoader(
        TensorDataset(torch.zeros(54_000, 1), torch.zeros(54_000, dtype=torch.long)),
        batch_size=256,
    )
    args = SimpleNamespace(
        method="vanilla-clip",
        epsilon_mode="calibrate",
        target_epsilon=2.0,
        delta=1e-5,
        epochs=30,
        accountant="rdp",
        epsilon_tolerance=1e-5,
        sigma=None,
    )

    metadata = run_exp._configure_noise_and_k(args, loader)

    assert len(loader) == 211
    assert captured["steps"] == 6_330
    assert "epochs" not in captured
    assert metadata["planned_logical_steps"] == 6_330


def test_budget_guard_allows_only_the_last_compliant_release():
    q = 1.0 / 211.0
    sigma = 1.1
    delta = 1e-5
    target = run_exp._epsilon_at_steps(
        accountant="rdp", sigma=sigma, sample_rate=q, steps=9, delta=delta
    )
    args = SimpleNamespace(
        epsilon_mode="calibrate",
        target_epsilon=target,
        accountant="rdp",
        delta=delta,
    )
    engine = SimpleNamespace(sample_rate=q)
    optimizer = SimpleNamespace(noise_multiplier=sigma)

    guard = run_exp._compute_privacy_budget_guard(
        args=args,
        privacy_engine=engine,
        optimizer=optimizer,
        sampling_metadata={"planned_logical_steps": 10},
    )

    assert guard["max_compliant_steps"] == 9
    assert guard["steps_truncated_by_plan"] == 1
    assert guard["planned_epsilon"] > target
    assert guard["projected_next_epsilon"] > target


def test_calibrated_guard_rejects_multi_step_mismatch():
    q = 1.0 / 211.0
    sigma = 1.1
    delta = 1e-5
    target = run_exp._epsilon_at_steps(
        accountant="rdp", sigma=sigma, sample_rate=q, steps=7, delta=delta
    )
    args = SimpleNamespace(
        epsilon_mode="calibrate",
        target_epsilon=target,
        accountant="rdp",
        delta=delta,
    )

    with pytest.raises(RuntimeError, match="releases would need to be removed"):
        run_exp._compute_privacy_budget_guard(
            args=args,
            privacy_engine=SimpleNamespace(sample_rate=q),
            optimizer=SimpleNamespace(noise_multiplier=sigma),
            sampling_metadata={"planned_logical_steps": 10},
        )
