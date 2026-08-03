from __future__ import annotations

import copy
import importlib
import itertools
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from slaclip.args import paper_k_upper_bound
from slaclip.protocol import (
    MAIN_BATCH_POOL,
    MAIN_BUDGETS,
    MAIN_C0_POOL,
    MAIN_LR_POOL,
    MAIN_SCHEDULE_POOL,
    PAPER_EPOCHS,
)
from tools.main_grid_common import (
    GridValidationError,
    PAPER_DATASETS,
    PAPER_METHODS,
    expected_candidate_count,
    generate_main_candidates,
    select_best_candidates,
    validate_complete_grid,
)

SOURCE_GIT = {
    "slaclip": {"commit": "a" * 40, "dirty": False},
    "opacus": {"commit": "b" * 40, "dirty": False},
}


@pytest.fixture(scope="module")
def full_grid():
    return generate_main_candidates(SOURCE_GIT)


def _resolved_method(candidate, sigma, K, expected_batch_size):
    method = candidate["group"]["method"]
    C0 = candidate["hyperparameters"]["C0"]
    classes = {
        "slaclip": "SlaClipOptimizer",
        "slaclip-q": "SlaClipQOptimizer",
        "vanilla-clip": "DPOptimizer",
        "adap-clip": "AdaClipDPOptimizer",
        "dc-sgd-e": "DCSGDEOptimizer",
        "autoclip": "AutoClipDPOptimizer",
    }
    resolved = {
        "method": method,
        "private": True,
        "optimizer_class": classes[method],
        "loss_reduction": "mean",
        "accountant_noise_multiplier": sigma,
        "initial_C0": C0,
        "c_min": 0.1,
        "c_max": 20.0,
    }
    if method in {"slaclip", "slaclip-q"}:
        resolved.update(
            {
                "K": K,
                "eta": 0.2,
                "beta": 0.5,
                "gamma": 0.5,
                "lambda_initial": C0 / K**0.5,
                "joint_release": "one isotropic d+K Gaussian vector",
                "controller": (
                    "Appendix C Eq.28-30" if method == "slaclip" else "Eq.12 median"
                ),
            }
        )
    elif method == "adap-clip":
        count_std = max(1.0, candidate["hyperparameters"]["batch_size"] / 20.0)
        gradient_sigma = (sigma**-2 - (2.0 * count_std) ** -2) ** -0.5
        resolved.update(
            {
                "target_unclipped_quantile": 0.5,
                "clipbound_learning_rate": 0.2,
                "count_noise_std": count_std,
                "gradient_noise_multiplier": gradient_sigma,
                "privacy_composition": "Andrew et al. Theorem 1 split",
                "count_query_kind": "centered_bit",
                "count_query_definition": "1{||g_i||<=C}-1/2",
                "count_query_l2_sensitivity": 0.5,
                "count_release_denominator": (
                    "expected_batch_size*accumulated_iterations"
                ),
                "count_release_expected_batch_size": expected_batch_size,
                "empty_poisson_count_release": True,
            }
        )
    elif method == "dc-sgd-e":
        resolved.update(
            {
                "histogram_noise_std": 5.0,
                "gradient_noise_multiplier": (sigma**-2 - 5.0**-2) ** -0.5,
                "percentile": 0.3,
                "stride_initial": 1.0,
                "bin_count": 20,
                "dimension": 12345,
                "nominal_batch_size": candidate["hyperparameters"]["batch_size"],
                "boundary_search_iteration_cap": 32,
                "empty_poisson_zero_histogram_release": True,
                "empty_poisson_histogram_input": "all-zero vector",
            }
        )
    elif method == "autoclip":
        resolved.update(
            {
                "mode": "auto_s",
                "auto_s_gamma": 0.01,
                "sensitivity_radius_R": C0,
                "transform": "R*g/(||g||+gamma), without Abadi clamp",
            }
        )
    elif method == "vanilla-clip":
        resolved["clipping"] = "global flat clipping"
    return resolved


def _payload(candidate, accuracy, *, test_accuracy=None, dirty=False):
    configuration = copy.deepcopy(candidate["expected_configuration"])
    sigma = 1.23
    private_train_size = 2 * int(candidate["hyperparameters"]["batch_size"]) - 1
    steps_per_epoch = 2
    sample_rate = 1.0 / steps_per_epoch
    expected_batch_size = int(private_train_size * sample_rate)
    K_max = paper_k_upper_bound(expected_batch_size, sigma)
    K = int(K_max)
    if candidate["group"]["method"] in {"slaclip", "slaclip-q"}:
        configuration["K"] = K
        K_selection = {
            "source": "Eq.14 floor",
            "K": K,
            "K_max_99": K_max,
            "within_eq14_bound": True,
            "expected_batch_size_used": expected_batch_size,
            "sigma_used": sigma,
        }
    else:
        configuration["K"] = None
        K_selection = None
    configuration["sigma"] = sigma
    if candidate["group"]["method"] == "adap-clip":
        configuration["adap_count_noise_std"] = max(
            1.0, candidate["hyperparameters"]["batch_size"] / 20.0
        )
    planned_steps = steps_per_epoch * PAPER_EPOCHS[candidate["group"]["dataset"]]
    epochs = [
        {
            "validation_accuracy": float(accuracy),
            "test_loss": None,
            "test_accuracy": None,
            "logical_steps_completed": epoch * steps_per_epoch,
        }
        for epoch in range(1, PAPER_EPOCHS[candidate["group"]["dataset"]] + 1)
    ]
    git = {
        name: {
            "commit": identity["commit"],
            "dirty": bool(dirty),
            "status_porcelain": [" M file.py"] if dirty else [],
            "describe": "test-dirty" if dirty else "test-clean",
        }
        for name, identity in SOURCE_GIT.items()
    }
    return {
        "schema_version": 2,
        "metadata": {
            "configuration": configuration,
            "protocol": {
                "protocol": "main",
                "phase": "selection",
                "validation_convention": {
                    "paper_specified": False,
                    "fraction": 0.1,
                    "selection_seed": 2026,
                },
                "K_selection": K_selection,
                "resolved_method": _resolved_method(
                    candidate, sigma, K, expected_batch_size
                ),
            },
            "git": git,
            "sampling": {
                "logical_steps_per_epoch": steps_per_epoch,
                "runtime_logical_steps_per_epoch": steps_per_epoch,
                "planned_logical_steps": planned_steps,
                "effective_sample_rate": sample_rate,
                "sampler_sample_rate": sample_rate,
                "accountant_sample_rate": sample_rate,
                "expected_batch_size": expected_batch_size,
                "optimizer_expected_batch_size": expected_batch_size,
            },
            "data": {
                "official_train_size": private_train_size + 10,
                "private_train_size": private_train_size,
                "validation_size": 10,
                "test_size": 0,
                "selection_phase_test_loader_created": False,
                "validation_split": {
                    "seed": 2026,
                    "holdout_fraction": 0.1,
                    "train_indices_sha256": "train",
                    "holdout_indices_sha256": "holdout",
                },
            },
        },
        "epochs": epochs,
        "final": {
            "epochs_completed": len(epochs),
            "logical_steps_completed": planned_steps,
            "epsilon": candidate["group"]["target_epsilon"],
            "delta": configuration["delta"],
            "target_epsilon": candidate["group"]["target_epsilon"],
            "epsilon_mode": "calibrate",
            "sigma": sigma,
            "final_validation_accuracy": float(accuracy),
            "final_test_loss": None,
            "final_test_accuracy": test_accuracy,
            "privacy_budget_guard": {
                "enabled": True,
                "planned_steps": planned_steps,
                "max_compliant_steps": planned_steps,
                "planned_epsilon": candidate["group"]["target_epsilon"],
                "projected_next_epsilon": None,
                "completed_steps": planned_steps,
                "steps_truncated": 0,
                "last_compliant_epsilon": candidate["group"]["target_epsilon"],
                "stopped_before_release": False,
            },
        },
    }


def test_complete_grid_count_and_unique_run_names(full_grid):
    validate_complete_grid(full_grid)
    assert len(full_grid) == expected_candidate_count() == 16200
    assert len({candidate["run_name"] for candidate in full_grid}) == len(full_grid)
    assert all(candidate["selection_seed"] == 2026 for candidate in full_grid)
    assert all(
        "--acknowledge-public-validation" in candidate["command_argv"]
        for candidate in full_grid
    )
    fixed_flags = {
        "--workers": "4",
        "--device": "auto",
        "--data-root": "data",
        "--max-sequence-length": "256",
    }
    sample_argv = full_grid[0]["command_argv"]
    for flag, value in fixed_flags.items():
        assert sample_argv[sample_argv.index(flag) + 1] == value


def test_every_method_uses_the_same_dataset_grid(full_grid):
    for dataset in PAPER_DATASETS:
        expected = set(
            itertools.product(
                MAIN_LR_POOL,
                MAIN_BATCH_POOL[dataset],
                MAIN_C0_POOL,
                MAIN_SCHEDULE_POOL,
            )
        )
        pools = {}
        for method in PAPER_METHODS:
            pools[method] = {
                (
                    entry["hyperparameters"]["lr"],
                    entry["hyperparameters"]["batch_size"],
                    entry["hyperparameters"]["C0"],
                    entry["hyperparameters"]["lr_schedule"],
                )
                for entry in full_grid
                if entry["group"]["method"] == method
                and entry["group"]["dataset"] == dataset
                and entry["group"]["budget_index"] == 1
            }
            assert pools[method] == expected
        assert all(pool == pools[PAPER_METHODS[0]] for pool in pools.values())


def test_tie_break_is_explicit_and_stable(full_grid):
    group = [
        candidate
        for candidate in full_grid
        if candidate["group"]
        == {
            "method": "slaclip",
            "dataset": "mnist",
            "budget_index": 1,
            "target_epsilon": MAIN_BUDGETS["mnist"][0],
        }
    ]
    low = next(
        c
        for c in group
        if c["hyperparameters"]
        == {"lr": 0.01, "batch_size": 256, "C0": 0.1, "lr_schedule": "constant"}
    )
    high = next(
        c
        for c in group
        if c["hyperparameters"]
        == {"lr": 0.05, "batch_size": 256, "C0": 0.1, "lr_schedule": "constant"}
    )
    payloads = {
        low["candidate_id"]: _payload(low, 0.8),
        high["candidate_id"]: _payload(high, 0.8),
    }
    retrains = select_best_candidates([high, low], payloads)
    assert len(retrains) == 3
    assert {entry["seed"] for entry in retrains} == {42, 43, 44}
    assert all(
        entry["selected_candidate_id"] == low["candidate_id"] for entry in retrains
    )
    assert all(
        "final.final_validation_accuracy" in entry["selection_policy"]["metric"]
        for entry in retrains
    )


def test_selection_rejects_any_test_metric_instead_of_ranking_with_it(full_grid):
    candidate = next(
        entry
        for entry in full_grid
        if entry["group"]["method"] == "autoclip"
        and entry["group"]["dataset"] == "cifar10"
    )
    payload = _payload(candidate, 0.2, test_accuracy=0.99)
    with pytest.raises(
        GridValidationError, match="forbidden selection-time final_test_accuracy"
    ):
        select_best_candidates([candidate], {candidate["candidate_id"]: payload})


def test_selection_rejects_dirty_or_mismatched_metadata(full_grid):
    candidate = full_grid[0]
    dirty = _payload(candidate, 0.5, dirty=True)
    with pytest.raises(GridValidationError, match="dirty slaclip tree"):
        select_best_candidates([candidate], {candidate["candidate_id"]: dirty})

    mismatched = _payload(candidate, 0.5)
    mismatched["metadata"]["configuration"]["lr"] = 1.0
    with pytest.raises(GridValidationError, match="configuration mismatch for lr"):
        select_best_candidates([candidate], {candidate["candidate_id"]: mismatched})


def test_selection_rejects_missing_candidate_run(full_grid):
    with pytest.raises(GridValidationError, match="Missing run JSON"):
        select_best_candidates([full_grid[0]], {})


def test_selection_validates_resolved_mechanism_for_all_six_methods(full_grid):
    candidates = []
    for method in PAPER_METHODS:
        candidates.append(
            next(
                candidate
                for candidate in full_grid
                if candidate["group"]["method"] == method
                and candidate["group"]["dataset"] == "mnist"
                and candidate["group"]["budget_index"] == 1
            )
        )
    payloads = {
        candidate["candidate_id"]: _payload(candidate, 0.7) for candidate in candidates
    }
    retrains = select_best_candidates(candidates, payloads)
    assert len(retrains) == len(PAPER_METHODS) * 3


def test_selection_rejects_tampered_autoclip_resolution_and_fixed_config(full_grid):
    candidate = next(
        candidate
        for candidate in full_grid
        if candidate["group"]["method"] == "autoclip"
    )
    wrong_mechanism = _payload(candidate, 0.6)
    wrong_mechanism["metadata"]["protocol"]["resolved_method"]["auto_s_gamma"] = 0.5
    with pytest.raises(GridValidationError, match="auto_s_gamma"):
        select_best_candidates(
            [candidate], {candidate["candidate_id"]: wrong_mechanism}
        )

    wrong_sequence_length = _payload(candidate, 0.6)
    wrong_sequence_length["metadata"]["configuration"]["max_sequence_length"] = 128
    with pytest.raises(
        GridValidationError, match="configuration mismatch for max_sequence_length"
    ):
        select_best_candidates(
            [candidate], {candidate["candidate_id"]: wrong_sequence_length}
        )


def test_selection_rejects_final_sigma_disagreement(full_grid):
    candidate = full_grid[0]
    payload = _payload(candidate, 0.6)
    payload["final"]["sigma"] = 1.24
    with pytest.raises(GridValidationError, match="configuration sigma"):
        select_best_candidates([candidate], {candidate["candidate_id"]: payload})


def test_selection_accepts_only_a_proven_final_release_budget_fallback(full_grid):
    candidate = next(
        entry
        for entry in full_grid
        if entry["group"]["method"] == "vanilla-clip"
        and entry["group"]["dataset"] == "mnist"
    )
    payload = _payload(candidate, 0.6)
    target = float(candidate["group"]["target_epsilon"])
    planned = int(payload["final"]["logical_steps_completed"])
    actual = target - 0.05
    projected = target + 0.001
    payload["epochs"][-1]["logical_steps_completed"] = planned - 1
    payload["final"]["logical_steps_completed"] = planned - 1
    payload["final"]["epsilon"] = actual
    payload["final"]["privacy_budget_guard"].update(
        {
            "max_compliant_steps": planned - 1,
            "planned_epsilon": projected,
            "projected_next_epsilon": projected,
            "completed_steps": planned - 1,
            "steps_truncated": 1,
            "last_compliant_epsilon": actual,
            "stopped_before_release": True,
        }
    )

    retrains = select_best_candidates(
        [candidate], {candidate["candidate_id"]: payload}
    )
    assert len(retrains) == 3

    over_budget = copy.deepcopy(payload)
    over_budget["final"]["epsilon"] = target + 1e-4
    over_budget["final"]["privacy_budget_guard"]["last_compliant_epsilon"] = (
        target + 1e-4
    )
    with pytest.raises(GridValidationError, match="exceeds target"):
        select_best_candidates(
            [candidate], {candidate["candidate_id"]: over_budget}
        )

    unproven = copy.deepcopy(payload)
    unproven["final"]["privacy_budget_guard"]["projected_next_epsilon"] = target
    unproven["final"]["privacy_budget_guard"]["planned_epsilon"] = target
    with pytest.raises(GridValidationError, match="does not prove"):
        select_best_candidates([candidate], {candidate["candidate_id"]: unproven})


def test_resolved_metadata_reads_noise_and_clip_from_optimizer_instance(monkeypatch):
    # Import the runner without importing optional dataset stacks: this unit only
    # exercises mechanism metadata and must remain a lightweight CPU test.
    data_stub = ModuleType("slaclip.data")
    data_stub.make_dataloaders = object()
    monkeypatch.setitem(sys.modules, "slaclip.data", data_stub)
    monkeypatch.delitem(sys.modules, "run_exp", raising=False)
    run_exp = importlib.import_module("run_exp")

    optimizer_type = type("DPOptimizer", (), {})
    optimizer = optimizer_type()
    optimizer.noise_multiplier = 1.25
    optimizer.max_grad_norm = 0.75
    optimizer.loss_reduction = "mean"
    args = SimpleNamespace(
        method="vanilla-clip",
        sigma=9.0,
        C0=9.0,
        c_min=0.1,
        c_max=20.0,
    )
    resolved = run_exp._resolved_method_configuration(args, optimizer)
    assert resolved["accountant_noise_multiplier"] == 1.25
    assert resolved["initial_C0"] == 0.75
    monkeypatch.delitem(sys.modules, "run_exp", raising=False)


def test_selection_rejects_tampered_empty_batch_query_semantics(full_grid):
    adap = next(
        candidate
        for candidate in full_grid
        if candidate["group"]["method"] == "adap-clip"
    )
    adap_payload = _payload(adap, 0.6)
    adap_payload["metadata"]["protocol"]["resolved_method"][
        "count_query_kind"
    ] = "uncentered_count"
    with pytest.raises(GridValidationError, match="count_query_kind"):
        select_best_candidates([adap], {adap["candidate_id"]: adap_payload})

    dc = next(
        candidate
        for candidate in full_grid
        if candidate["group"]["method"] == "dc-sgd-e"
    )
    dc_payload = _payload(dc, 0.6)
    dc_payload["metadata"]["protocol"]["resolved_method"][
        "empty_poisson_zero_histogram_release"
    ] = False
    with pytest.raises(
        GridValidationError, match="empty_poisson_zero_histogram_release"
    ):
        select_best_candidates([dc], {dc["candidate_id"]: dc_payload})


def test_one_winner_per_method_dataset_budget_generates_270_retrains(full_grid):
    candidates_by_group = {}
    for candidate in full_grid:
        key = (
            candidate["group"]["method"],
            candidate["group"]["dataset"],
            candidate["group"]["budget_index"],
        )
        candidates_by_group.setdefault(key, candidate)
    candidates = list(candidates_by_group.values())
    assert len(candidates) == len(PAPER_METHODS) * len(PAPER_DATASETS) * 3 == 90
    payloads = {
        candidate["candidate_id"]: _payload(candidate, 0.7) for candidate in candidates
    }
    retrains = select_best_candidates(candidates, payloads)
    assert len(retrains) == 270
