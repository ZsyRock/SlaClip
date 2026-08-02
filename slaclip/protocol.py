from __future__ import annotations

import math
from argparse import Namespace
from typing import Any

CAMERA_READY_REVISION = "ICML 2026 camera-ready, 2026-05-29"
MAIN_BUDGETS = {
    "cifar10": (4.0, 6.0, 8.0),
    "mnist": (1.0, 2.0, 3.0),
    "fmnist": (1.0, 2.0, 3.0),
    "imdb": (2.0, 4.0, 6.0),
    "names": (1.0, 2.0, 3.0),
}
CONTROLLED_BUDGETS = {
    "cifar10": (5.0, 7.0, 9.0),
    "mnist": (1.0, 2.0, 3.0),
    "fmnist": (1.0, 2.0, 3.0),
    "imdb": (2.0, 4.0, 6.0),
    "names": (2.0, 4.0, 5.0),
}
PAPER_EPOCHS = {
    "cifar10": 90,
    "mnist": 30,
    "fmnist": 30,
    "imdb": 90,
    "names": 30,
}
PAPER_DELTA = {
    "cifar10": 1e-5,
    "mnist": 1e-5,
    "fmnist": 1e-5,
    "imdb": 1e-5,
    "names": 8e-5,
}
CONTROLLED_BATCH = {
    "cifar10": 1024,
    "mnist": 512,
    "fmnist": 512,
    "imdb": 256,
    "names": 512,
}
MAIN_BATCH_POOL = {
    "cifar10": (512, 1024, 2048),
    "mnist": (256, 512, 1024),
    "fmnist": (256, 512, 1024),
    "imdb": (256, 512, 1024),
    "names": (256, 512, 1024),
}
MAIN_LR_POOL = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
MAIN_C0_POOL = (0.1, 0.5, 1.0, 5.0, 10.0)
MAIN_SCHEDULE_POOL = ("constant", "cos")
RETRAIN_SEEDS = (42, 43, 44)

# The camera-ready paper says "a single selection seed" and "validation
# accuracy", but does not disclose the seed, split ratio, split indices, or
# whether the holdout reduces the private training N. These constants are an
# explicit deterministic convention, not a recovered paper setting.
DEFAULT_SELECTION_SEED = 2026
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_DATA_SPLIT_SEED = 2026


def _float_in(value: float, candidates: tuple[float, ...]) -> bool:
    return any(
        math.isclose(float(value), x, rel_tol=0.0, abs_tol=1e-12) for x in candidates
    )


def _paper_budget(args: Namespace, budgets: dict[str, tuple[float, ...]]) -> None:
    candidates = budgets[str(args.dataset)]
    if args.budget_index is None and args.target_epsilon is None:
        raise ValueError(
            f"--protocol {args.protocol} requires exactly one paper budget: pass "
            "--budget-index {1,2,3} or --target-epsilon."
        )

    if args.budget_index is not None:
        indexed = float(candidates[int(args.budget_index) - 1])
        if args.target_epsilon is not None and not math.isclose(
            float(args.target_epsilon), indexed, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"--budget-index {args.budget_index} means epsilon={indexed:g} for "
                f"{args.dataset}/{args.protocol}, but --target-epsilon={args.target_epsilon:g}."
            )
        args.target_epsilon = indexed
    else:
        target = float(args.target_epsilon)
        if not _float_in(target, candidates):
            raise ValueError(
                f"epsilon={target:g} is not a {args.protocol} paper budget for {args.dataset}: "
                f"{list(candidates)}"
            )
        args.budget_index = next(
            i + 1
            for i, candidate in enumerate(candidates)
            if math.isclose(target, candidate, rel_tol=0.0, abs_tol=1e-12)
        )


def _require_or_set(args: Namespace, name: str, expected: Any, context: str) -> None:
    actual = getattr(args, name)
    if actual is None:
        setattr(args, name, expected)
        return
    if isinstance(expected, float):
        equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(
            f"{context} requires --{name.replace('_', '-')}={expected}, got {actual}. "
            "Use --protocol custom for a deliberate deviation."
        )


def apply_protocol(args: Namespace) -> dict[str, Any]:
    """Resolve and validate a single auditable experiment configuration.

    This function mutates ``args`` only after argparse has established types.
    It returns protocol metadata suitable for the immutable run record.
    """

    dataset = str(args.dataset)
    method = str(args.method)

    if args.use_paper_budgets:
        raise ValueError(
            "--use-paper-budgets was ambiguous and is no longer supported. Choose "
            "--protocol main or --protocol controlled and pass --budget-index."
        )
    if args.calibrate_sigma:
        if args.epsilon_mode not in (None, "calibrate"):
            raise ValueError("--calibrate-sigma conflicts with --epsilon-mode")
        args.epsilon_mode = "calibrate"

    if args.c_min >= args.c_max:
        raise ValueError("--c-min must be smaller than --c-max")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.momentum is not None and args.momentum < 0:
        raise ValueError("--momentum must be >= 0")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")

    if args.slot_fb_beta is not None:
        args.beta = float(args.slot_fb_beta)

    paper_protocol = args.protocol in {"main", "controlled"}
    if paper_protocol and args.accountant != "rdp":
        raise ValueError("Camera-ready paper presets require --accountant rdp")
    if paper_protocol and args.optim != "SGD":
        raise ValueError("Camera-ready paper presets require --optim SGD")
    if paper_protocol and args.grad_sample_mode != "hooks":
        raise ValueError("Camera-ready paper presets require --grad-sample-mode hooks")
    if paper_protocol and not bool(args.strict_paper_check):
        raise ValueError("Camera-ready paper presets require --strict-paper-check")
    if paper_protocol and not (
        math.isclose(float(args.c_min), 0.1, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(args.c_max), 20.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            "Camera-ready presets fix the engineering C guardrails to [0.1, 20]"
        )
    if (
        paper_protocol
        and method == "slaclip"
        and not math.isclose(float(args.beta), 0.5, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Camera-ready SlaClip fixes beta=0.5")
    if (
        paper_protocol
        and method in {"slaclip", "slaclip-q", "adap-clip"}
        and not math.isclose(float(args.gamma), 0.5, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"Camera-ready {method} fixes gamma=0.5")
    if paper_protocol and dataset == "names":
        names_requirements = {
            "embedding_dim": 128,
            "hidden_size": 128,
            "n_layers": 1,
            "dropout": 0.0,
            "rnn_arch": "lstm",
            "bidirectional": False,
        }
        for name, expected in names_requirements.items():
            _require_or_set(args, name, expected, "Paper Names DP-LSTM")
    if paper_protocol and method == "dc-sgd-e":
        dc_expected = {
            "dc_percentile": 0.3,
            "dc_stride": 1.0,
            "dc_bin_count": 20,
            "dc_histogram_std": 5.0,
        }
        for name, expected in dc_expected.items():
            _require_or_set(args, name, expected, "Paper DC-SGD-E")

    if args.protocol == "main":
        if method == "nondp":
            raise ValueError(
                "Use --protocol custom for the single non-private reference run"
            )
        if method in {"slaclip", "slaclip-q"} and args.K is not None:
            raise ValueError(
                "Main-paper SlaClip chooses K from Eq. (14) after sigma calibration; "
                "omit --K. Use --protocol controlled/custom for an explicit K ablation."
            )
        if args.phase is None:
            raise ValueError(
                "--protocol main requires --phase selection or --phase retrain"
            )
        if args.phase not in {"selection", "retrain"}:
            raise ValueError(
                "--protocol main supports only selection and retrain phases"
            )
        _paper_budget(args, MAIN_BUDGETS)
        _require_or_set(args, "epochs", PAPER_EPOCHS[dataset], "Main paper protocol")
        _require_or_set(args, "delta", PAPER_DELTA[dataset], "Main paper protocol")
        if args.sigma is not None:
            raise ValueError(
                "Main paper protocol calibrates sigma for the full horizon; omit --sigma."
            )
        if args.epsilon_mode not in (None, "calibrate"):
            raise ValueError("Main paper protocol requires --epsilon-mode calibrate")
        args.epsilon_mode = "calibrate"
        _require_or_set(
            args,
            "epsilon_tolerance",
            1e-5,
            "Main paper Table 2 noise calibration",
        )

        if (
            args.batch_size is None
            or int(args.batch_size) not in MAIN_BATCH_POOL[dataset]
        ):
            raise ValueError(
                f"Main paper protocol requires an explicit --batch-size from "
                f"{list(MAIN_BATCH_POOL[dataset])}."
            )
        if args.lr is None or not _float_in(float(args.lr), MAIN_LR_POOL):
            raise ValueError(
                f"Main paper protocol requires an explicit --lr from {list(MAIN_LR_POOL)}."
            )
        if args.lr_schedule not in MAIN_SCHEDULE_POOL:
            raise ValueError(
                "Main paper protocol requires explicit --lr-schedule constant or cos."
            )
        if dataset == "names":
            _require_or_set(args, "momentum", 0.0, "Main Names recipe")
            _require_or_set(args, "weight_decay", 0.0, "Main Names recipe")
        else:
            _require_or_set(args, "momentum", 0.9, "Main paper recipe")
            _require_or_set(args, "weight_decay", 5e-4, "Main paper recipe")

        if args.C0 is None or not _float_in(float(args.C0), MAIN_C0_POOL):
            raise ValueError(
                f"Main paper protocol requires an explicit --C0 from {list(MAIN_C0_POOL)}."
            )

        if args.phase == "selection":
            if not bool(args.acknowledge_public_validation):
                raise ValueError(
                    "Main selection uses validation accuracy outside the DP mechanism. Pass "
                    "--acknowledge-public-validation to confirm that the privacy guarantee "
                    "covers only the reduced private-train subset, not the holdout, and "
                    "does not compose the full multi-candidate sweep."
                )
            if args.seed is not None and int(args.seed) != int(args.selection_seed):
                raise ValueError(
                    "Main selection must use the single --selection-seed for every candidate."
                )
            args.seed = int(args.selection_seed)
        else:
            if args.seed is None:
                raise ValueError(
                    "Main retrain requires one explicit --seed from {42,43,44}; run all three."
                )
            if int(args.seed) not in RETRAIN_SEEDS:
                raise ValueError(f"Main retrain seed must be one of {RETRAIN_SEEDS}")

    elif args.protocol == "controlled":
        if method == "nondp":
            raise ValueError("Controlled private budgets do not apply to nondp")
        if args.phase not in (None, "controlled"):
            raise ValueError("--protocol controlled requires --phase controlled")
        args.phase = "controlled"
        _paper_budget(args, CONTROLLED_BUDGETS)
        _require_or_set(
            args, "epochs", PAPER_EPOCHS[dataset], "Controlled paper protocol"
        )
        _require_or_set(
            args, "delta", PAPER_DELTA[dataset], "Controlled paper protocol"
        )
        _require_or_set(
            args, "batch_size", CONTROLLED_BATCH[dataset], "Controlled paper protocol"
        )
        _require_or_set(args, "C0", 1.0, "Controlled paper protocol")
        if args.epsilon_mode not in (None, "hard-stop"):
            raise ValueError(
                "Controlled Appendix F fixes sigma=1 and therefore requires hard-stop semantics."
            )
        args.epsilon_mode = "hard-stop"
        _require_or_set(args, "sigma", 1.0, "Controlled paper protocol")
        if args.seed is None:
            raise ValueError(
                "Controlled comparison requires one explicit --seed from {42,43,44}; "
                "run all three."
            )
        if int(args.seed) not in RETRAIN_SEEDS:
            raise ValueError(f"Controlled seed must be one of {RETRAIN_SEEDS}")

        if dataset == "names":
            _require_or_set(args, "lr", 2.0, "Controlled Names recipe")
            _require_or_set(args, "momentum", 0.0, "Controlled Names recipe")
            _require_or_set(args, "weight_decay", 0.0, "Controlled Names recipe")
            _require_or_set(args, "lr_schedule", "constant", "Controlled Names recipe")
        else:
            _require_or_set(args, "lr", 0.1, "Controlled fixed recipe")
            _require_or_set(args, "momentum", 0.9, "Controlled fixed recipe")
            _require_or_set(args, "weight_decay", 5e-4, "Controlled fixed recipe")
            _require_or_set(args, "lr_schedule", "cos", "Controlled fixed recipe")
        if method in {"slaclip", "slaclip-q"}:
            _require_or_set(args, "eta", 0.5, "Controlled SlaClip recipe")

    else:
        if args.phase is None:
            args.phase = "custom"
        if args.epochs is None:
            args.epochs = PAPER_EPOCHS[dataset]
        if args.delta is None:
            args.delta = PAPER_DELTA[dataset]
        if args.batch_size is None:
            args.batch_size = CONTROLLED_BATCH[dataset]
        if args.lr is None:
            args.lr = 2.0 if dataset == "names" else 0.1
        if args.momentum is None:
            args.momentum = 0.0 if dataset == "names" else 0.9
        if args.weight_decay is None:
            args.weight_decay = 0.0 if dataset == "names" else 5e-4
        if args.lr_schedule is None:
            args.lr_schedule = "constant" if dataset == "names" else "cos"
        if args.C0 is None:
            args.C0 = 1.0
        if args.sigma is None:
            args.sigma = 1.0
        if args.seed is None:
            args.seed = 42
        if args.epsilon_mode is None:
            args.epsilon_mode = "none"
        if (
            args.epsilon_mode in {"calibrate", "hard-stop"}
            and args.target_epsilon is None
        ):
            raise ValueError(
                f"--epsilon-mode {args.epsilon_mode} requires --target-epsilon"
            )
        if args.epsilon_mode == "none" and args.target_epsilon is not None:
            raise ValueError(
                "--target-epsilon has no implicit meaning. Select --epsilon-mode calibrate "
                "or hard-stop."
            )

    if args.epsilon_tolerance is None:
        # Upstream Opacus default for custom/non-calibrated commands. Keeping it
        # explicit in the immutable config avoids version-dependent ambiguity.
        args.epsilon_tolerance = 0.01

    if args.eta is None:
        args.eta = 0.2
    if args.batch_size_test is None:
        args.batch_size_test = (
            1024 if dataset in {"mnist", "fmnist"} else args.batch_size
        )
    if args.max_physical_batch_size is None:
        if method == "nondp":
            args.max_physical_batch_size = int(args.batch_size)
        else:
            conservative_cap = 32 if dataset in {"imdb", "names"} else 64
            args.max_physical_batch_size = min(int(args.batch_size), conservative_cap)
    if int(args.max_physical_batch_size) > int(args.batch_size):
        args.max_physical_batch_size = int(args.batch_size)
    if method == "nondp" and int(args.max_physical_batch_size) != int(args.batch_size):
        raise ValueError(
            "Physical microbatch splitting is implemented through Opacus and is only valid "
            "for private methods; set it equal to --batch-size for nondp."
        )
    if not float(args.c_min) <= float(args.C0) <= float(args.c_max):
        raise ValueError(
            "--C0 must lie within the configured [--c-min, --c-max] guardrails"
        )
    if args.phase == "selection" and not bool(args.acknowledge_public_validation):
        raise ValueError(
            "Selection phase requires --acknowledge-public-validation for its privacy scope"
        )
    return {
        "camera_ready_revision": CAMERA_READY_REVISION,
        "protocol": str(args.protocol),
        "phase": str(args.phase),
        "paper_protocol": bool(paper_protocol),
        "budget_family": (
            "Table 1 fairly tuned"
            if args.protocol == "main"
            else (
                "Appendix F controlled fixed recipe"
                if args.protocol == "controlled"
                else "custom"
            )
        ),
        "validation_convention": {
            "paper_specified": False,
            "fraction": float(args.validation_fraction),
            "selection_seed": int(args.selection_seed),
            "note": (
                "The public camera-ready paper does not state the validation ratio, split "
                "indices, or selection-seed value. Selection holds out this fraction from "
                "the official training split and recalibrates privacy for the reduced N; "
                "retrain restores the full official training split. Validation is a public "
                "non-DP tuning query: the selection-run DP guarantee covers only the reduced "
                "private-train subset, not the holdout. Each command accounts one candidate; "
                "privacy across a full hyperparameter sweep is not composed by this runner."
            ),
        },
        "names_split_convention": {
            "paper_specified": False,
            "train_fraction": 0.9,
            "seed": int(args.data_split_seed),
            "note": "Fixed independently of training seed so seeds 42/43/44 share one test set.",
        },
        "fixed_training_convention": {
            "optimizer": str(args.optim),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "note": (
                "The main grid enumerates lr, logical batch, C0, and schedule. Optimizer, "
                "momentum, and weight decay are fixed to the released/default paper recipe "
                "rather than treated as additional grid dimensions."
            ),
        },
        "execution_scope": {
            "single_candidate_command": True,
            "published_shared_grid_enumeration_automated": True,
            "published_shared_grid_candidate_count": 16200,
            "selection_metric": "final-epoch public validation accuracy",
            "retrain_generation_automated": True,
            "unpublished_method_specific_grids_reconstructed": False,
            "note": (
                "run_exp.py validates and executes one candidate. Companion tools under "
                "tools/ generate the complete published shared grid, reject incomplete or "
                "test-contaminated results, rank final validation accuracy, and generate "
                "seeds 42/43/44 retrains. The paper does not publish every method-specific "
                "adaptive-parameter range, so those dimensions are not invented."
            ),
        },
    }
