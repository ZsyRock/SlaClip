from __future__ import annotations

import json
import math
import shlex
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from slaclip.args import paper_k_upper_bound
from slaclip.protocol import (
    DEFAULT_DATA_SPLIT_SEED,
    DEFAULT_SELECTION_SEED,
    DEFAULT_VALIDATION_FRACTION,
    MAIN_BATCH_POOL,
    MAIN_BUDGETS,
    MAIN_C0_POOL,
    MAIN_LR_POOL,
    MAIN_SCHEDULE_POOL,
    PAPER_DELTA,
    PAPER_EPOCHS,
    RETRAIN_SEEDS,
)

MANIFEST_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 2
PAPER_METHODS = (
    "slaclip",
    "slaclip-q",
    "vanilla-clip",
    "adap-clip",
    "dc-sgd-e",
    "autoclip",
)
PAPER_DATASETS = tuple(MAIN_BUDGETS)
SELECTION_OUTPUT_DIR = "outputs/main_selection"
RETRAIN_OUTPUT_DIR = "outputs/main_retrain"


class GridValidationError(ValueError):
    """Raised when a grid manifest or selection result is not auditable."""


def _float_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def candidate_id(
    *,
    method: str,
    dataset: str,
    budget_index: int,
    lr: float,
    batch_size: int,
    C0: float,
    lr_schedule: str,
) -> str:
    return (
        f"main-select-{method}-{dataset}-b{int(budget_index)}"
        f"-lr{_float_slug(lr)}-bs{int(batch_size)}"
        f"-c{_float_slug(C0)}-{lr_schedule}-s{DEFAULT_SELECTION_SEED}"
    )


def retrain_id(candidate: Mapping[str, Any], seed: int) -> str:
    group = candidate["group"]
    hp = candidate["hyperparameters"]
    return (
        f"main-retrain-{group['method']}-{group['dataset']}"
        f"-b{int(group['budget_index'])}-lr{_float_slug(hp['lr'])}"
        f"-bs{int(hp['batch_size'])}-c{_float_slug(hp['C0'])}"
        f"-{hp['lr_schedule']}-s{int(seed)}"
    )


def git_snapshot(repo: Path, *, ignore_submodules: bool = False) -> dict[str, Any]:
    """Return the exact clean Git identity used to generate a manifest."""

    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GridValidationError(
                f"Cannot inspect Git repository {repo}: {exc}"
            ) from exc
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status_args = ["status", "--porcelain=v1"]
    if ignore_submodules:
        status_args.append("--ignore-submodules=all")
    status = run(*status_args)
    if not commit:
        raise GridValidationError(f"Git repository {repo} has no HEAD commit")
    if status:
        raise GridValidationError(
            f"Refusing to generate a paper manifest from dirty repository {repo}:\n{status}"
        )
    return {"commit": commit, "dirty": False}


def source_git_identity(*, slaclip_root: Path, opacus_root: Path) -> dict[str, Any]:
    return {
        "slaclip": git_snapshot(slaclip_root),
        # The nested SlaClip commit is tracked independently; do not require a
        # parent Opacus commit merely to update that gitlink.
        "opacus": git_snapshot(opacus_root, ignore_submodules=True),
    }


def _normalise_source_git(source_git: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for repo_name in ("slaclip", "opacus"):
        raw = source_git.get(repo_name)
        if not isinstance(raw, Mapping):
            raise GridValidationError(f"source_git.{repo_name} must be an object")
        commit = raw.get("commit")
        dirty = raw.get("dirty")
        if not isinstance(commit, str) or not commit.strip():
            raise GridValidationError(f"source_git.{repo_name}.commit is missing")
        if dirty is not False:
            raise GridValidationError(
                f"source_git.{repo_name}.dirty must be exactly false"
            )
        result[repo_name] = {"commit": commit.strip(), "dirty": False}
    return result


def _fixed_configuration(
    *,
    method: str,
    dataset: str,
    budget_index: int,
    lr: float,
    batch_size: int,
    C0: float,
    lr_schedule: str,
    run_name: str,
    output_dir: str,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    target_epsilon = float(MAIN_BUDGETS[dataset][budget_index - 1])
    batch_size_test = 1024 if dataset in {"mnist", "fmnist"} else batch_size
    physical_cap = 32 if dataset in {"imdb", "names"} else 64
    return {
        "method": method,
        "dataset": dataset,
        "protocol": "main",
        "phase": phase,
        "budget_index": int(budget_index),
        "target_epsilon": target_epsilon,
        "epsilon_mode": "calibrate",
        "epsilon_tolerance": 1e-5,
        "epochs": int(PAPER_EPOCHS[dataset]),
        "delta": float(PAPER_DELTA[dataset]),
        "seed": int(seed),
        "selection_seed": int(DEFAULT_SELECTION_SEED),
        "data_split_seed": int(DEFAULT_DATA_SPLIT_SEED),
        "validation_fraction": float(DEFAULT_VALIDATION_FRACTION),
        "acknowledge_public_validation": phase == "selection",
        "batch_size": int(batch_size),
        "batch_size_test": int(batch_size_test),
        "max_physical_batch_size": min(int(batch_size), physical_cap),
        "workers": 4,
        "device": "auto",
        "optim": "SGD",
        "lr": float(lr),
        "momentum": 0.0 if dataset == "names" else 0.9,
        "weight_decay": 0.0 if dataset == "names" else 5e-4,
        "lr_schedule": lr_schedule,
        "accountant": "rdp",
        "secure_mode": False,
        "grad_sample_mode": "hooks",
        "C0": float(C0),
        "eta": 0.2,
        "beta": 0.5,
        "gamma": 0.5,
        "c_min": 0.1,
        "c_max": 20.0,
        "slot_fb_beta": None,
        "strict_paper_check": True,
        "dc_percentile": 0.3,
        "dc_stride": 1.0,
        "dc_bin_count": 20,
        "dc_histogram_std": 5.0,
        "calibrate_sigma": False,
        "use_paper_budgets": False,
        "embedding_dim": 128,
        "hidden_size": 128,
        "n_layers": 1,
        "dropout": 0.0,
        "rnn_arch": "lstm",
        "bidirectional": False,
        "max_sequence_length": 256,
        "run_name": run_name,
        "out_dir": output_dir,
        "data_root": "data",
        "overwrite": False,
    }


def _command_argv(configuration: Mapping[str, Any]) -> list[str]:
    argv = [
        "python",
        "SlaClip/run_exp.py",
        "--method",
        str(configuration["method"]),
        "--dataset",
        str(configuration["dataset"]),
        "--protocol",
        "main",
        "--phase",
        str(configuration["phase"]),
        "--budget-index",
        str(configuration["budget_index"]),
        "--seed",
        str(configuration["seed"]),
        "--selection-seed",
        str(configuration["selection_seed"]),
        "--data-split-seed",
        str(configuration["data_split_seed"]),
        "--validation-fraction",
        format(float(configuration["validation_fraction"]), ".12g"),
        "--epochs",
        str(configuration["epochs"]),
        "--delta",
        format(float(configuration["delta"]), ".12g"),
        "--batch-size",
        str(configuration["batch_size"]),
        "--batch-size-test",
        str(configuration["batch_size_test"]),
        "--max-physical-batch-size",
        str(configuration["max_physical_batch_size"]),
        "--workers",
        str(configuration["workers"]),
        "--device",
        str(configuration["device"]),
        "--optim",
        "SGD",
        "--lr",
        format(float(configuration["lr"]), ".12g"),
        "--momentum",
        format(float(configuration["momentum"]), ".12g"),
        "--weight-decay",
        format(float(configuration["weight_decay"]), ".12g"),
        "--lr-schedule",
        str(configuration["lr_schedule"]),
        "--accountant",
        "rdp",
        "--epsilon-tolerance",
        "0.00001",
        "--grad-sample-mode",
        "hooks",
        "--C0",
        format(float(configuration["C0"]), ".12g"),
        "--eta",
        "0.2",
        "--beta",
        "0.5",
        "--gamma",
        "0.5",
        "--c-min",
        "0.1",
        "--c-max",
        "20",
        "--dc-percentile",
        format(float(configuration["dc_percentile"]), ".12g"),
        "--dc-stride",
        format(float(configuration["dc_stride"]), ".12g"),
        "--dc-bin-count",
        str(configuration["dc_bin_count"]),
        "--dc-histogram-std",
        format(float(configuration["dc_histogram_std"]), ".12g"),
        "--max-sequence-length",
        str(configuration["max_sequence_length"]),
        "--embedding-dim",
        str(configuration["embedding_dim"]),
        "--hidden-size",
        str(configuration["hidden_size"]),
        "--n-layers",
        str(configuration["n_layers"]),
        "--dropout",
        format(float(configuration["dropout"]), ".12g"),
        "--rnn-arch",
        str(configuration["rnn_arch"]),
        "--strict-paper-check",
        "--no-secure-mode",
        "--run-name",
        str(configuration["run_name"]),
        "--out-dir",
        str(configuration["out_dir"]),
        "--data-root",
        str(configuration["data_root"]),
    ]
    if configuration["phase"] == "selection":
        argv.append("--acknowledge-public-validation")
    return argv


def generate_main_candidates(
    source_git: Mapping[str, Any],
    *,
    output_dir: str = SELECTION_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    """Generate the complete Table-1 shared grid without executing it."""

    clean_git = _normalise_source_git(source_git)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for method in PAPER_METHODS:
        for dataset in PAPER_DATASETS:
            for budget_index, target_epsilon in enumerate(
                MAIN_BUDGETS[dataset], start=1
            ):
                grid_index = 0
                for lr in MAIN_LR_POOL:
                    for batch_size in MAIN_BATCH_POOL[dataset]:
                        for C0 in MAIN_C0_POOL:
                            for schedule_index, lr_schedule in enumerate(
                                MAIN_SCHEDULE_POOL
                            ):
                                run_name = candidate_id(
                                    method=method,
                                    dataset=dataset,
                                    budget_index=budget_index,
                                    lr=lr,
                                    batch_size=batch_size,
                                    C0=C0,
                                    lr_schedule=lr_schedule,
                                )
                                if run_name in seen_ids:
                                    raise AssertionError(
                                        f"Duplicate run name: {run_name}"
                                    )
                                seen_ids.add(run_name)
                                configuration = _fixed_configuration(
                                    method=method,
                                    dataset=dataset,
                                    budget_index=budget_index,
                                    lr=lr,
                                    batch_size=batch_size,
                                    C0=C0,
                                    lr_schedule=lr_schedule,
                                    run_name=run_name,
                                    output_dir=output_dir,
                                    phase="selection",
                                    seed=DEFAULT_SELECTION_SEED,
                                )
                                argv = _command_argv(configuration)
                                records.append(
                                    {
                                        "schema_version": MANIFEST_SCHEMA_VERSION,
                                        "record_type": "main_selection_candidate",
                                        "candidate_id": run_name,
                                        "run_name": run_name,
                                        "run_json": f"{run_name}.json",
                                        "group": {
                                            "method": method,
                                            "dataset": dataset,
                                            "budget_index": budget_index,
                                            "target_epsilon": float(target_epsilon),
                                        },
                                        "hyperparameters": {
                                            "lr": float(lr),
                                            "batch_size": int(batch_size),
                                            "C0": float(C0),
                                            "lr_schedule": lr_schedule,
                                        },
                                        "selection_seed": DEFAULT_SELECTION_SEED,
                                        "grid_index_within_group": grid_index,
                                        "tie_break_key": [
                                            float(lr),
                                            int(batch_size),
                                            float(C0),
                                            schedule_index,
                                            run_name,
                                        ],
                                        "source_git": clean_git,
                                        "expected_configuration": configuration,
                                        "command_argv": argv,
                                        "command": shlex.join(argv),
                                    }
                                )
                                grid_index += 1
    return records


def expected_candidate_count() -> int:
    per_budget = (
        len(MAIN_LR_POOL)
        * len(next(iter(MAIN_BATCH_POOL.values())))
        * len(MAIN_C0_POOL)
        * len(MAIN_SCHEDULE_POOL)
    )
    return len(PAPER_METHODS) * len(PAPER_DATASETS) * 3 * per_budget


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _equal(actual: Any, expected: Any) -> bool:
    if _is_number(actual) and _is_number(expected):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def _need(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise GridValidationError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _need_mapping(
    mapping: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    value = _need(mapping, key, context)
    if not isinstance(value, Mapping):
        raise GridValidationError(f"{context}.{key} must be an object")
    return value


def _canonical_candidate_from_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    context = f"candidate {candidate.get('candidate_id', '<missing>')}"
    if _need(candidate, "schema_version", context) != MANIFEST_SCHEMA_VERSION:
        raise GridValidationError(f"{context} has an unsupported schema version")
    if _need(candidate, "record_type", context) != "main_selection_candidate":
        raise GridValidationError(f"{context} has an invalid record_type")
    group = _need_mapping(candidate, "group", context)
    hp = _need_mapping(candidate, "hyperparameters", context)
    method = _need(group, "method", f"{context}.group")
    dataset = _need(group, "dataset", f"{context}.group")
    budget_index = _need(group, "budget_index", f"{context}.group")
    if method not in PAPER_METHODS or dataset not in PAPER_DATASETS:
        raise GridValidationError(f"{context} has an unknown method or dataset")
    if type(budget_index) is not int or budget_index not in (1, 2, 3):
        raise GridValidationError(f"{context} has an invalid budget_index")
    lr = _need(hp, "lr", f"{context}.hyperparameters")
    batch_size = _need(hp, "batch_size", f"{context}.hyperparameters")
    C0 = _need(hp, "C0", f"{context}.hyperparameters")
    lr_schedule = _need(hp, "lr_schedule", f"{context}.hyperparameters")
    if not any(_equal(lr, value) for value in MAIN_LR_POOL):
        raise GridValidationError(f"{context} lr is outside the shared paper pool")
    if type(batch_size) is not int or batch_size not in MAIN_BATCH_POOL[dataset]:
        raise GridValidationError(
            f"{context} batch size is outside the shared paper pool"
        )
    if not any(_equal(C0, value) for value in MAIN_C0_POOL):
        raise GridValidationError(f"{context} C0 is outside the shared paper pool")
    if lr_schedule not in MAIN_SCHEDULE_POOL:
        raise GridValidationError(
            f"{context} schedule is outside the shared paper pool"
        )

    expected_id = candidate_id(
        method=method,
        dataset=dataset,
        budget_index=budget_index,
        lr=float(lr),
        batch_size=batch_size,
        C0=float(C0),
        lr_schedule=lr_schedule,
    )
    if _need(candidate, "candidate_id", context) != expected_id:
        raise GridValidationError(f"{context} candidate_id does not match its fields")
    if _need(candidate, "run_name", context) != expected_id:
        raise GridValidationError(f"{context} run_name does not match its fields")
    if _need(candidate, "run_json", context) != f"{expected_id}.json":
        raise GridValidationError(f"{context} run_json does not match its run_name")
    expected_target = float(MAIN_BUDGETS[dataset][budget_index - 1])
    if not _equal(_need(group, "target_epsilon", f"{context}.group"), expected_target):
        raise GridValidationError(
            f"{context} target epsilon does not match the paper budget"
        )

    clean_git = _normalise_source_git(_need_mapping(candidate, "source_git", context))
    expected_configuration = _fixed_configuration(
        method=method,
        dataset=dataset,
        budget_index=budget_index,
        lr=float(lr),
        batch_size=batch_size,
        C0=float(C0),
        lr_schedule=lr_schedule,
        run_name=expected_id,
        output_dir=str(
            _need_mapping(candidate, "expected_configuration", context).get(
                "out_dir", SELECTION_OUTPUT_DIR
            )
        ),
        phase="selection",
        seed=DEFAULT_SELECTION_SEED,
    )
    supplied_configuration = _need_mapping(candidate, "expected_configuration", context)
    for key, expected in expected_configuration.items():
        actual = _need(supplied_configuration, key, f"{context}.expected_configuration")
        if not _equal(actual, expected):
            raise GridValidationError(
                f"{context}.expected_configuration.{key}={actual!r}, expected {expected!r}"
            )
    if _need(candidate, "selection_seed", context) != DEFAULT_SELECTION_SEED:
        raise GridValidationError(f"{context} selection seed must be 2026")
    lr_index = next(i for i, value in enumerate(MAIN_LR_POOL) if _equal(lr, value))
    batch_index = MAIN_BATCH_POOL[dataset].index(batch_size)
    C0_index = next(i for i, value in enumerate(MAIN_C0_POOL) if _equal(C0, value))
    schedule_index = MAIN_SCHEDULE_POOL.index(lr_schedule)
    expected_grid_index = (
        (lr_index * len(MAIN_BATCH_POOL[dataset]) + batch_index) * len(MAIN_C0_POOL)
        + C0_index
    ) * len(MAIN_SCHEDULE_POOL) + schedule_index
    if _need(candidate, "grid_index_within_group", context) != expected_grid_index:
        raise GridValidationError(f"{context} has an invalid within-group grid index")
    expected_tie_key = [
        float(lr),
        int(batch_size),
        float(C0),
        schedule_index,
        expected_id,
    ]
    if _need(candidate, "tie_break_key", context) != expected_tie_key:
        raise GridValidationError(f"{context} has an invalid tie-break key")
    expected_argv = _command_argv(expected_configuration)
    if _need(candidate, "command_argv", context) != expected_argv:
        raise GridValidationError(
            f"{context} command_argv does not match its configuration"
        )
    if _need(candidate, "command", context) != shlex.join(expected_argv):
        raise GridValidationError(f"{context} command does not match command_argv")
    return {
        "candidate": candidate,
        "group": group,
        "hyperparameters": hp,
        "source_git": clean_git,
        "expected_configuration": expected_configuration,
    }


def validate_complete_grid(candidates: Sequence[Mapping[str, Any]]) -> None:
    if len(candidates) != expected_candidate_count():
        raise GridValidationError(
            f"Selection manifest has {len(candidates)} candidates; expected "
            f"{expected_candidate_count()} for the complete shared grid"
        )
    seen: set[str] = set()
    counts: dict[tuple[str, str, int], int] = {}
    source_identity: str | None = None
    for candidate in candidates:
        canonical = _canonical_candidate_from_record(candidate)
        candidate_name = str(candidate["candidate_id"])
        if candidate_name in seen:
            raise GridValidationError(f"Duplicate candidate_id: {candidate_name}")
        seen.add(candidate_name)
        group = canonical["group"]
        key = (group["method"], group["dataset"], group["budget_index"])
        counts[key] = counts.get(key, 0) + 1
        serialised_source = json.dumps(canonical["source_git"], sort_keys=True)
        if source_identity is None:
            source_identity = serialised_source
        elif serialised_source != source_identity:
            raise GridValidationError("Candidates do not share one exact Git identity")
    expected_per_group = (
        len(MAIN_LR_POOL)
        * len(next(iter(MAIN_BATCH_POOL.values())))
        * len(MAIN_C0_POOL)
        * len(MAIN_SCHEDULE_POOL)
    )
    expected_groups = {
        (method, dataset, budget_index)
        for method in PAPER_METHODS
        for dataset in PAPER_DATASETS
        for budget_index in (1, 2, 3)
    }
    if set(counts) != expected_groups or any(
        count != expected_per_group for count in counts.values()
    ):
        raise GridValidationError("Selection manifest is not the complete shared grid")


def _validate_run_git(
    metadata: Mapping[str, Any], candidate: Mapping[str, Any], context: str
) -> None:
    run_git = _need_mapping(metadata, "git", f"{context}.metadata")
    expected_git = _need_mapping(candidate, "source_git", context)
    for repo_name in ("slaclip", "opacus"):
        snapshot = _need_mapping(run_git, repo_name, f"{context}.metadata.git")
        expected = _need_mapping(expected_git, repo_name, f"{context}.source_git")
        if _need(snapshot, "dirty", f"{context}.metadata.git.{repo_name}") is not False:
            raise GridValidationError(
                f"{context} was run from a dirty {repo_name} tree"
            )
        status = _need(
            snapshot, "status_porcelain", f"{context}.metadata.git.{repo_name}"
        )
        if status != []:
            raise GridValidationError(
                f"{context} has non-empty {repo_name} status_porcelain"
            )
        commit = _need(snapshot, "commit", f"{context}.metadata.git.{repo_name}")
        if commit != _need(expected, "commit", f"{context}.source_git.{repo_name}"):
            raise GridValidationError(
                f"{context} {repo_name} commit does not match manifest"
            )
        describe = _need(snapshot, "describe", f"{context}.metadata.git.{repo_name}")
        if not isinstance(describe, str) or describe.endswith("-dirty"):
            raise GridValidationError(
                f"{context} has invalid dirty Git describe metadata"
            )


def _validate_no_test_query(
    metadata: Mapping[str, Any],
    epochs: Sequence[Any],
    final: Mapping[str, Any],
    context: str,
) -> None:
    data = _need_mapping(metadata, "data", f"{context}.metadata")
    if (
        _need(data, "selection_phase_test_loader_created", f"{context}.metadata.data")
        is not False
    ):
        raise GridValidationError(f"{context} did not prove test-loader isolation")
    if _need(data, "test_size", f"{context}.metadata.data") != 0:
        raise GridValidationError(f"{context} exposed a test set during selection")
    for key in ("final_test_loss", "final_test_accuracy"):
        if _need(final, key, f"{context}.final") is not None:
            raise GridValidationError(
                f"{context} contains forbidden selection-time {key}"
            )
    for index, epoch in enumerate(epochs, start=1):
        if not isinstance(epoch, Mapping):
            raise GridValidationError(
                f"{context}.epochs[{index - 1}] must be an object"
            )
        for key in ("test_loss", "test_accuracy"):
            if _need(epoch, key, f"{context}.epochs[{index - 1}]") is not None:
                raise GridValidationError(
                    f"{context} contains forbidden selection-time epoch {key}"
                )


def _require_resolved_value(
    resolved: Mapping[str, Any], key: str, expected: Any, context: str
) -> Any:
    actual = _need(resolved, key, context)
    if not _equal(actual, expected):
        raise GridValidationError(
            f"{context}.{key}={actual!r}, expected resolved value {expected!r}"
        )
    return actual


def _validate_resolved_method(
    *,
    metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    configuration: Mapping[str, Any],
    canonical: Mapping[str, Any],
    final: Mapping[str, Any],
    context: str,
) -> None:
    """Cross-check the instantiated mechanism, not merely its generic CLI fields."""

    method = str(canonical["group"]["method"])
    resolved = _need_mapping(
        protocol, "resolved_method", f"{context}.metadata.protocol"
    )
    resolved_context = f"{context}.metadata.protocol.resolved_method"
    final_sigma = _need(final, "sigma", f"{context}.final")
    if (
        not _is_number(final_sigma)
        or not math.isfinite(float(final_sigma))
        or float(final_sigma) <= 0
    ):
        raise GridValidationError(f"{context}.final.sigma must be finite and positive")
    final_sigma = float(final_sigma)

    configured_sigma = _need(
        configuration, "sigma", f"{context}.metadata.configuration"
    )
    if not _equal(configured_sigma, final_sigma):
        raise GridValidationError(
            f"{context} configuration sigma does not match final.sigma"
        )

    optimizer_classes = {
        "slaclip": "SlaClipOptimizer",
        "slaclip-q": "SlaClipQOptimizer",
        "vanilla-clip": "DPOptimizer",
        "adap-clip": "AdaClipDPOptimizer",
        "dc-sgd-e": "DCSGDEOptimizer",
        "autoclip": "AutoClipDPOptimizer",
    }
    for key, expected in (
        ("method", method),
        ("private", True),
        ("optimizer_class", optimizer_classes[method]),
        ("loss_reduction", "mean"),
        ("accountant_noise_multiplier", final_sigma),
        ("initial_C0", canonical["hyperparameters"]["C0"]),
        ("c_min", 0.1),
        ("c_max", 20.0),
    ):
        _require_resolved_value(resolved, key, expected, resolved_context)

    K_selection = _need(protocol, "K_selection", f"{context}.metadata.protocol")
    configured_K = _need(configuration, "K", f"{context}.metadata.configuration")
    if method in {"slaclip", "slaclip-q"}:
        if type(configured_K) is not int or configured_K <= 0:
            raise GridValidationError(f"{context} has no valid resolved K")
        k_metadata = K_selection
        if not isinstance(k_metadata, Mapping):
            raise GridValidationError(f"{context} is missing Eq.14 K metadata")
        sampling = _need_mapping(metadata, "sampling", f"{context}.metadata")
        expected_batch_size = _need(
            sampling, "expected_batch_size", f"{context}.metadata.sampling"
        )
        if type(expected_batch_size) is not int or expected_batch_size <= 0:
            raise GridValidationError(
                f"{context} sampling.expected_batch_size must be a positive integer"
            )
        computed_upper = paper_k_upper_bound(expected_batch_size, final_sigma)
        for key, expected in (
            ("source", "Eq.14 floor"),
            ("K", configured_K),
            ("within_eq14_bound", True),
            ("expected_batch_size_used", expected_batch_size),
            ("sigma_used", final_sigma),
            ("K_max_99", computed_upper),
        ):
            actual = _need(k_metadata, key, f"{context}.metadata.protocol.K_selection")
            if not _equal(actual, expected):
                raise GridValidationError(
                    f"{context} K-selection metadata mismatch for {key}"
                )
        if configured_K != math.floor(computed_upper):
            raise GridValidationError(f"{context} K is not the floor of Eq.14")
        for key, expected in (
            ("K", configured_K),
            ("eta", 0.2),
            ("beta", 0.5),
            ("gamma", 0.5),
            (
                "lambda_initial",
                float(canonical["hyperparameters"]["C0"]) / math.sqrt(configured_K),
            ),
            ("joint_release", "one isotropic d+K Gaussian vector"),
            (
                "controller",
                "Appendix C Eq.28-30" if method == "slaclip" else "Eq.12 median",
            ),
        ):
            _require_resolved_value(resolved, key, expected, resolved_context)
    else:
        if configured_K is not None or K_selection is not None:
            raise GridValidationError(
                f"{context} non-SlaClip method unexpectedly records K selection"
            )

    if method == "adap-clip":
        count_noise_std = max(
            1.0, float(canonical["hyperparameters"]["batch_size"]) / 20.0
        )
        sampling = _need_mapping(metadata, "sampling", f"{context}.metadata")
        expected_batch_size = _need(
            sampling, "expected_batch_size", f"{context}.metadata.sampling"
        )
        if type(expected_batch_size) is not int or expected_batch_size <= 0:
            raise GridValidationError(
                f"{context} sampling.expected_batch_size must be a positive integer"
            )
        configured_count_std = _need(
            configuration,
            "adap_count_noise_std",
            f"{context}.metadata.configuration",
        )
        if not _equal(configured_count_std, count_noise_std):
            raise GridValidationError(
                f"{context} AdaClip count-noise configuration mismatch"
            )
        inv_gradient_variance = final_sigma**-2 - (2.0 * count_noise_std) ** -2
        if inv_gradient_variance <= 0:
            raise GridValidationError(f"{context} has an invalid AdaClip privacy split")
        gradient_sigma = inv_gradient_variance**-0.5
        for key, expected in (
            ("target_unclipped_quantile", 0.5),
            ("clipbound_learning_rate", 0.2),
            ("count_noise_std", count_noise_std),
            ("gradient_noise_multiplier", gradient_sigma),
            ("privacy_composition", "Andrew et al. Theorem 1 split"),
            ("count_query_kind", "centered_bit"),
            ("count_query_definition", "1{||g_i||<=C}-1/2"),
            ("count_query_l2_sensitivity", 0.5),
            (
                "count_release_denominator",
                "expected_batch_size*accumulated_iterations",
            ),
            ("count_release_expected_batch_size", expected_batch_size),
            ("empty_poisson_count_release", True),
        ):
            _require_resolved_value(resolved, key, expected, resolved_context)
    elif method == "dc-sgd-e":
        histogram_sigma = 5.0
        inv_gradient_variance = final_sigma**-2 - histogram_sigma**-2
        if inv_gradient_variance <= 0:
            raise GridValidationError(
                f"{context} has an invalid DC-SGD-E privacy split"
            )
        gradient_sigma = inv_gradient_variance**-0.5
        for key, expected in (
            ("histogram_noise_std", histogram_sigma),
            ("gradient_noise_multiplier", gradient_sigma),
            ("percentile", 0.3),
            ("stride_initial", 1.0),
            ("bin_count", 20),
            ("nominal_batch_size", canonical["hyperparameters"]["batch_size"]),
            ("boundary_search_iteration_cap", 32),
            ("empty_poisson_zero_histogram_release", True),
            ("empty_poisson_histogram_input", "all-zero vector"),
        ):
            _require_resolved_value(resolved, key, expected, resolved_context)
        dimension = _need(resolved, "dimension", resolved_context)
        if type(dimension) is not int or dimension <= 0:
            raise GridValidationError(
                f"{resolved_context}.dimension must be a positive integer"
            )
    elif method == "autoclip":
        for key, expected in (
            ("mode", "auto_s"),
            ("auto_s_gamma", 0.01),
            ("sensitivity_radius_R", canonical["hyperparameters"]["C0"]),
            ("transform", "R*g/(||g||+gamma), without Abadi clamp"),
        ):
            _require_resolved_value(resolved, key, expected, resolved_context)
    elif method == "vanilla-clip":
        _require_resolved_value(
            resolved, "clipping", "global flat clipping", resolved_context
        )


def _validate_private_horizon(
    *,
    metadata: Mapping[str, Any],
    epochs: Sequence[Any],
    final: Mapping[str, Any],
    expected_configuration: Mapping[str, Any],
    target_epsilon: float,
    expected_epochs: int,
    context: str,
) -> None:
    actual_epsilon = float(_need(final, "epsilon", f"{context}.final"))
    tolerance = float(expected_configuration["epsilon_tolerance"])
    if actual_epsilon > target_epsilon + 1e-12:
        raise GridValidationError(
            f"{context}.final.epsilon={actual_epsilon} exceeds target "
            f"{target_epsilon}; calibration tolerance is not privacy slack"
        )

    sampling = _need_mapping(metadata, "sampling", f"{context}.metadata")
    private_train_size = _need(
        _need_mapping(metadata, "data", f"{context}.metadata"),
        "private_train_size",
        f"{context}.metadata.data",
    )
    batch_size = int(expected_configuration["batch_size"])
    expected_steps_per_epoch = -(-int(private_train_size) // batch_size)
    planned_steps = expected_steps_per_epoch * expected_epochs
    expected_sample_rate = 1.0 / float(expected_steps_per_epoch)
    expected_batch_size = int(int(private_train_size) * expected_sample_rate)
    for key, expected in (
        ("logical_steps_per_epoch", expected_steps_per_epoch),
        ("runtime_logical_steps_per_epoch", expected_steps_per_epoch),
        ("planned_logical_steps", planned_steps),
        ("effective_sample_rate", expected_sample_rate),
        ("sampler_sample_rate", expected_sample_rate),
        ("accountant_sample_rate", expected_sample_rate),
        ("expected_batch_size", expected_batch_size),
        ("optimizer_expected_batch_size", expected_batch_size),
    ):
        if not _equal(_need(sampling, key, f"{context}.metadata.sampling"), expected):
            raise GridValidationError(
                f"{context}.metadata.sampling.{key} does not match the fixed "
                "Poisson mechanism"
            )

    guard = _need_mapping(final, "privacy_budget_guard", f"{context}.final")
    if _need(guard, "enabled", f"{context}.final.privacy_budget_guard") is not True:
        raise GridValidationError(f"{context} did not enable the privacy-budget guard")
    completed_steps = _need(final, "logical_steps_completed", f"{context}.final")
    if type(completed_steps) is not int:
        raise GridValidationError(f"{context}.final logical step count must be an integer")
    steps_truncated = planned_steps - completed_steps
    if steps_truncated not in (0, 1):
        raise GridValidationError(
            f"{context} completed {completed_steps}/{planned_steps} releases; only a "
            "single final-release privacy fallback is admissible"
        )
    for key, expected in (
        ("planned_steps", planned_steps),
        ("completed_steps", completed_steps),
        ("steps_truncated", steps_truncated),
        ("max_compliant_steps", completed_steps),
        ("last_compliant_epsilon", actual_epsilon),
    ):
        if not _equal(_need(guard, key, f"{context}.final.privacy_budget_guard"), expected):
            raise GridValidationError(
                f"{context}.final.privacy_budget_guard.{key} is inconsistent"
            )

    if steps_truncated == 0:
        if target_epsilon - actual_epsilon > 2.0 * tolerance:
            raise GridValidationError(
                f"{context}.final.epsilon={actual_epsilon} is farther than the "
                f"declared calibration tolerance from target {target_epsilon}"
            )
        if _need(
            guard, "stopped_before_release", f"{context}.final.privacy_budget_guard"
        ) is not False:
            raise GridValidationError(f"{context} falsely reports a budget stop")
        expected_epoch_steps = [
            expected_steps_per_epoch * epoch for epoch in range(1, expected_epochs + 1)
        ]
    else:
        projected_next = _need(
            guard, "projected_next_epsilon", f"{context}.final.privacy_budget_guard"
        )
        if not _is_number(projected_next) or not target_epsilon < float(projected_next):
            raise GridValidationError(
                f"{context} does not prove that the omitted final release exceeded budget"
            )
        if _need(
            guard, "stopped_before_release", f"{context}.final.privacy_budget_guard"
        ) is not True:
            raise GridValidationError(
                f"{context} did not stop before the over-budget release"
            )
        if not _equal(
            _need(guard, "planned_epsilon", f"{context}.final.privacy_budget_guard"),
            projected_next,
        ):
            raise GridValidationError(
                f"{context} projected final-release epsilon is inconsistent"
            )
        expected_epoch_steps = [
            expected_steps_per_epoch * epoch for epoch in range(1, expected_epochs)
        ] + [planned_steps - 1]

    observed_epoch_steps = [
        _need(epoch, "logical_steps_completed", f"{context}.epochs")
        for epoch in epochs
    ]
    if observed_epoch_steps != expected_epoch_steps:
        raise GridValidationError(
            f"{context} epoch logical-step trajectory does not match its privacy horizon"
        )


def validate_selection_run(
    candidate: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[float, str]:
    """Validate one run and return its only permitted selection metric/fingerprint."""

    canonical = _canonical_candidate_from_record(candidate)
    context = f"run {candidate['candidate_id']}"
    if not isinstance(payload, Mapping):
        raise GridValidationError(f"{context} JSON root must be an object")
    if _need(payload, "schema_version", context) != RUN_SCHEMA_VERSION:
        raise GridValidationError(f"{context} has an unsupported run schema version")
    metadata = _need_mapping(payload, "metadata", context)
    configuration = _need_mapping(metadata, "configuration", f"{context}.metadata")
    for key, expected in canonical["expected_configuration"].items():
        actual = _need(configuration, key, f"{context}.metadata.configuration")
        if not _equal(actual, expected):
            raise GridValidationError(
                f"{context} configuration mismatch for {key}: "
                f"got {actual!r}, expected {expected!r}"
            )

    protocol = _need_mapping(metadata, "protocol", f"{context}.metadata")
    if _need(protocol, "protocol", f"{context}.metadata.protocol") != "main":
        raise GridValidationError(f"{context} protocol metadata is not main")
    if _need(protocol, "phase", f"{context}.metadata.protocol") != "selection":
        raise GridValidationError(f"{context} protocol metadata is not selection")
    convention = _need_mapping(
        protocol, "validation_convention", f"{context}.metadata.protocol"
    )
    for key, expected in (
        ("paper_specified", False),
        ("fraction", DEFAULT_VALIDATION_FRACTION),
        ("selection_seed", DEFAULT_SELECTION_SEED),
    ):
        actual = _need(
            convention, key, f"{context}.metadata.protocol.validation_convention"
        )
        if not _equal(actual, expected):
            raise GridValidationError(
                f"{context} validation convention mismatch for {key}"
            )

    _validate_run_git(metadata, candidate, context)
    epochs = _need(payload, "epochs", context)
    if not isinstance(epochs, list):
        raise GridValidationError(f"{context}.epochs must be an array")
    final = _need_mapping(payload, "final", context)
    _validate_no_test_query(metadata, epochs, final, context)

    expected_epochs = int(canonical["expected_configuration"]["epochs"])
    if len(epochs) != expected_epochs:
        raise GridValidationError(
            f"{context} has {len(epochs)} epoch records; expected {expected_epochs}"
        )
    if _need(final, "epochs_completed", f"{context}.final") != expected_epochs:
        raise GridValidationError(f"{context} did not complete the prescribed horizon")
    metric = _need(final, "final_validation_accuracy", f"{context}.final")
    if not _is_number(metric) or not math.isfinite(float(metric)):
        raise GridValidationError(f"{context} has no finite final_validation_accuracy")
    metric = float(metric)
    if not 0.0 <= metric <= 1.0:
        raise GridValidationError(f"{context} validation accuracy must be in [0, 1]")
    last_metric = _need(epochs[-1], "validation_accuracy", f"{context}.epochs[-1]")
    if not _equal(last_metric, metric):
        raise GridValidationError(
            f"{context} final_validation_accuracy does not match the final epoch"
        )
    for key, expected in (
        ("epsilon_mode", "calibrate"),
        ("target_epsilon", canonical["group"]["target_epsilon"]),
        ("delta", canonical["expected_configuration"]["delta"]),
    ):
        actual = _need(final, key, f"{context}.final")
        if not _equal(actual, expected):
            raise GridValidationError(f"{context} final metadata mismatch for {key}")
    for key in ("epsilon", "sigma"):
        value = _need(final, key, f"{context}.final")
        if (
            not _is_number(value)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise GridValidationError(
                f"{context}.final.{key} must be finite and positive"
            )
    _validate_resolved_method(
        metadata=metadata,
        protocol=protocol,
        configuration=configuration,
        canonical=canonical,
        final=final,
        context=context,
    )
    _validate_private_horizon(
        metadata=metadata,
        epochs=epochs,
        final=final,
        expected_configuration=canonical["expected_configuration"],
        target_epsilon=float(canonical["group"]["target_epsilon"]),
        expected_epochs=expected_epochs,
        context=context,
    )

    data_fingerprint = json.dumps(
        _need_mapping(metadata, "data", f"{context}.metadata"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return metric, data_fingerprint


def validate_retrain_run(
    record: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[float, float]:
    """Validate one final three-seed run, including its exact DP horizon."""

    context = f"retrain {_need(record, 'run_name', 'retrain record')}"
    if _need(record, "record_type", context) != "main_retrain_run":
        raise GridValidationError(f"{context} has an invalid record_type")
    if _need(payload, "schema_version", context) != RUN_SCHEMA_VERSION:
        raise GridValidationError(f"{context} has an unsupported run schema")
    metadata = _need_mapping(payload, "metadata", context)
    configuration = _need_mapping(metadata, "configuration", f"{context}.metadata")
    expected_configuration = _need_mapping(record, "expected_configuration", context)
    for key, expected in expected_configuration.items():
        if not _equal(_need(configuration, key, f"{context}.configuration"), expected):
            raise GridValidationError(f"{context} configuration mismatch for {key}")
    protocol = _need_mapping(metadata, "protocol", f"{context}.metadata")
    if _need(protocol, "protocol", f"{context}.metadata.protocol") != "main":
        raise GridValidationError(f"{context} is not a main-protocol run")
    if _need(protocol, "phase", f"{context}.metadata.protocol") != "retrain":
        raise GridValidationError(f"{context} is not a retrain run")
    _validate_run_git(metadata, record, context)

    epochs = _need(payload, "epochs", context)
    final = _need_mapping(payload, "final", context)
    if not isinstance(epochs, list):
        raise GridValidationError(f"{context}.epochs must be an array")
    expected_epochs = int(expected_configuration["epochs"])
    if len(epochs) != expected_epochs:
        raise GridValidationError(f"{context} did not record {expected_epochs} epochs")
    if _need(final, "epochs_completed", f"{context}.final") != expected_epochs:
        raise GridValidationError(f"{context} did not complete the prescribed epochs")

    data = _need_mapping(metadata, "data", f"{context}.metadata")
    if _need(data, "validation_size", f"{context}.metadata.data") != 0:
        raise GridValidationError(f"{context} unexpectedly created a validation split")
    for index, epoch in enumerate(epochs, start=1):
        epoch = _need_mapping({"epoch": epoch}, "epoch", f"{context}.epochs[{index}]")
        for key in ("validation_loss", "validation_accuracy", "test_loss", "test_accuracy"):
            if epoch.get(key) is not None:
                raise GridValidationError(
                    f"{context}.epochs[{index}] unexpectedly publishes {key}"
                )
    if final.get("final_validation_accuracy") is not None:
        raise GridValidationError(f"{context} unexpectedly publishes validation accuracy")
    test_accuracy = _need(final, "final_test_accuracy", f"{context}.final")
    test_loss = _need(final, "final_test_loss", f"{context}.final")
    if not _is_number(test_accuracy) or not math.isfinite(float(test_accuracy)):
        raise GridValidationError(f"{context} has invalid final test accuracy")
    if not 0.0 <= float(test_accuracy) <= 1.0:
        raise GridValidationError(f"{context} final test accuracy is outside [0,1]")
    if not _is_number(test_loss) or not math.isfinite(float(test_loss)) or float(test_loss) < 0:
        raise GridValidationError(f"{context} has invalid final test loss")
    if _need(final, "epsilon_mode", f"{context}.final") != "calibrate":
        raise GridValidationError(f"{context} did not use calibrated epsilon mode")
    for key in ("epsilon", "sigma"):
        value = _need(final, key, f"{context}.final")
        if not _is_number(value) or not math.isfinite(float(value)) or float(value) <= 0:
            raise GridValidationError(f"{context}.final.{key} must be finite and positive")

    canonical = {
        "group": _need_mapping(record, "group", context),
        "hyperparameters": _need_mapping(record, "hyperparameters", context),
    }
    _validate_resolved_method(
        metadata=metadata,
        protocol=protocol,
        configuration=configuration,
        canonical=canonical,
        final=final,
        context=context,
    )
    _validate_private_horizon(
        metadata=metadata,
        epochs=epochs,
        final=final,
        expected_configuration=expected_configuration,
        target_epsilon=float(canonical["group"]["target_epsilon"]),
        expected_epochs=expected_epochs,
        context=context,
    )
    return float(test_accuracy), float(test_loss)


def select_best_candidates(
    candidates: Sequence[Mapping[str, Any]],
    run_payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select winners solely by final validation accuracy and emit retrain entries."""

    seen: set[str] = set()
    scored: dict[
        tuple[str, str, int], list[tuple[float, tuple[Any, ...], Mapping[str, Any]]]
    ] = {}
    data_fingerprints: dict[str, str] = {}
    common_git: str | None = None
    for candidate in candidates:
        canonical = _canonical_candidate_from_record(candidate)
        candidate_name = str(candidate["candidate_id"])
        if candidate_name in seen:
            raise GridValidationError(f"Duplicate candidate_id: {candidate_name}")
        seen.add(candidate_name)
        if candidate_name not in run_payloads:
            raise GridValidationError(
                f"Missing run JSON for candidate {candidate_name}"
            )
        metric, data_fingerprint = validate_selection_run(
            candidate, run_payloads[candidate_name]
        )
        dataset = str(canonical["group"]["dataset"])
        previous_fingerprint = data_fingerprints.setdefault(dataset, data_fingerprint)
        if previous_fingerprint != data_fingerprint:
            raise GridValidationError(
                f"Selection runs for {dataset} do not share identical data/split metadata"
            )
        serialised_git = json.dumps(canonical["source_git"], sort_keys=True)
        if common_git is None:
            common_git = serialised_git
        elif serialised_git != common_git:
            raise GridValidationError(
                "Selection candidates do not share one Git identity"
            )
        group_key = (
            canonical["group"]["method"],
            dataset,
            canonical["group"]["budget_index"],
        )
        hp = canonical["hyperparameters"]
        tie_key = (
            float(hp["lr"]),
            int(hp["batch_size"]),
            float(hp["C0"]),
            MAIN_SCHEDULE_POOL.index(str(hp["lr_schedule"])),
            candidate_name,
        )
        scored.setdefault(group_key, []).append((metric, tie_key, candidate))

    winners: list[dict[str, Any]] = []
    for group_key in sorted(scored):
        # Highest final validation accuracy wins. Exact ties use the ascending,
        # explicitly recorded (lr, batch, C0, schedule-order, candidate-id) key.
        ranked = sorted(scored[group_key], key=lambda item: (-item[0], item[1]))
        metric, tie_key, selected = ranked[0]
        hp = selected["hyperparameters"]
        group = selected["group"]
        for seed in RETRAIN_SEEDS:
            run_name = retrain_id(selected, seed)
            configuration = _fixed_configuration(
                method=str(group["method"]),
                dataset=str(group["dataset"]),
                budget_index=int(group["budget_index"]),
                lr=float(hp["lr"]),
                batch_size=int(hp["batch_size"]),
                C0=float(hp["C0"]),
                lr_schedule=str(hp["lr_schedule"]),
                run_name=run_name,
                output_dir=RETRAIN_OUTPUT_DIR,
                phase="retrain",
                seed=int(seed),
            )
            argv = _command_argv(configuration)
            winners.append(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "record_type": "main_retrain_run",
                    "run_name": run_name,
                    "run_json": f"{run_name}.json",
                    "group": dict(group),
                    "seed": int(seed),
                    "hyperparameters": dict(hp),
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_final_validation_accuracy": metric,
                    "selection_policy": {
                        "metric": "final.final_validation_accuracy",
                        "maximize": True,
                        "test_metrics": "forbidden and never read for ranking",
                        "tie_break": (
                            "ascending (lr, batch_size, C0, "
                            "schedule[constant,cos], candidate_id)"
                        ),
                        "winning_tie_break_key": list(tie_key),
                    },
                    "source_git": selected["source_git"],
                    "expected_configuration": configuration,
                    "command_argv": argv,
                    "command": shlex.join(argv),
                }
            )
    return winners


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GridValidationError(f"Cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GridValidationError(
                f"Invalid JSON on {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise GridValidationError(f"{path}:{line_number} is not a JSON object")
        records.append(record)
    if not records:
        raise GridValidationError(f"JSONL file {path} is empty")
    return records


class _RunPayloadDirectory(Mapping[str, Mapping[str, Any]]):
    """Read one potentially large run result at a time during selection."""

    def __init__(self, paths: Mapping[str, Path]):
        self._paths = dict(paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self):
        return iter(self._paths)

    def __contains__(self, key: object) -> bool:
        return key in self._paths

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        path = self._paths[key]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GridValidationError(f"Cannot read run JSON {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise GridValidationError(f"Run JSON {path} is not an object")
        return payload


def read_run_payloads(
    candidates: Sequence[Mapping[str, Any]], runs_dir: Path
) -> Mapping[str, Mapping[str, Any]]:
    missing: list[str] = []
    paths: dict[str, Path] = {}
    for candidate in candidates:
        candidate_name = str(_need(candidate, "candidate_id", "candidate"))
        filename = _need(candidate, "run_json", f"candidate {candidate_name}")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise GridValidationError(
                f"candidate {candidate_name} has an unsafe run_json filename"
            )
        path = runs_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        paths[candidate_name] = path
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise GridValidationError(
            f"Missing {len(missing)} required candidate run JSON files:\n{preview}{suffix}"
        )
    return _RunPayloadDirectory(paths)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_commands(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated only; running this file starts the paper protocol jobs.",
    ]
    lines.extend(str(record["command"]) for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o755)
