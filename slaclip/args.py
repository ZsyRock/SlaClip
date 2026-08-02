from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

# scipy.stats.norm.ppf(0.995). Kept as a literal so argument parsing does not
# add SciPy as a runtime dependency.
_Z_0995 = 2.5758293035489004
PAPER_PRACTICAL_K_SIGMA1 = {
    128: 8,
    256: 10,
    512: 20,
    1024: 30,
    2048: 50,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _fraction_001_099(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.01 <= parsed <= 0.99:
        raise argparse.ArgumentTypeError("must be in the inclusive range [0.01, 0.99]")
    return parsed


def paper_k_upper_bound(expected_batch_size: int, sigma: float) -> float:
    """Eq. (14)/(36) 99%-SNR upper bound from the camera-ready paper."""

    expected_batch_size = int(expected_batch_size)
    sigma = float(sigma)
    if expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be > 0")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0")
    return (expected_batch_size / (2.0 * _Z_0995 * sigma)) ** (2.0 / 3.0)


def paper_recommended_k(expected_batch_size: int, sigma: float = 1.0) -> int:
    """Take the largest integer satisfying the paper's Eq. (14) bound.

    The camera-ready Table 3 also lists convenient nearby values. Automatic
    selection deliberately follows the equation rather than that sigma=1-only
    lookup table, because main-table runs calibrate a different sigma for every
    dataset/batch/privacy-budget combination.
    """

    upper_bound = paper_k_upper_bound(expected_batch_size, sigma)
    selected = int(math.floor(upper_bound))
    if selected < 1:
        raise ValueError(
            "Eq. (14) has no admissible positive integer K for "
            f"expected_batch_size={int(expected_batch_size)}, sigma={float(sigma):.8g} "
            f"(K_max={upper_bound:.8g}). Increase the logical batch size or reduce sigma."
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable SlaClip camera-ready paper reproduction runner"
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "slaclip",
            "slaclip-q",
            "vanilla-clip",
            "adap-clip",
            "dc-sgd-e",
            "autoclip",
            "nondp",
        ],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "fmnist", "cifar10", "imdb", "names"],
    )
    parser.add_argument(
        "--protocol",
        default="custom",
        choices=["custom", "main", "controlled"],
        help=(
            "main=Table 1 fairly-tuned protocol; controlled=Appendix F fixed recipe; "
            "custom=no claim of exact paper protocol."
        ),
    )
    parser.add_argument(
        "--phase",
        default=None,
        choices=["selection", "retrain", "controlled", "custom"],
        help=(
            "Main-paper selection uses a deterministic training holdout; retrain uses the "
            "full official training set and evaluates test only at the end."
        ),
    )
    parser.add_argument(
        "--budget-index",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="One-based index into the selected paper protocol's three epsilon budgets.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--data-split-seed",
        type=int,
        default=2026,
        help=(
            "Fixed Names train/test split seed. The paper does not publish this seed; "
            "2026 is an explicit deterministic engineering convention."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=_fraction_001_099,
        default=0.1,
        help=(
            "Training holdout used only in selection phase. The paper does not publish a "
            "ratio; 0.1 is an explicit engineering convention."
        ),
    )
    parser.add_argument(
        "--acknowledge-public-validation",
        action="store_true",
        help=(
            "Required for main selection: acknowledges that validation accuracy is a "
            "non-DP query, the accountant covers only the remaining private-train subset, "
            "and a full multi-candidate sweep is not composed by this single-run accountant."
        ),
    )
    parser.add_argument("--epochs", type=_positive_int, default=None)

    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--batch-size-test", type=_positive_int, default=None)
    parser.add_argument(
        "--max-physical-batch-size",
        type=_positive_int,
        default=None,
        help=(
            "Opacus BatchMemoryManager cap. Conservative 8GB defaults are 64 for vision "
            "and 32 for IMDB/Names; logical Poisson batch size/accounting are unchanged."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--optim", default="SGD", choices=["SGD", "Adam", "RMSprop"])
    parser.add_argument("--lr", type=_positive_float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-schedule", default=None, choices=["constant", "cos"])

    parser.add_argument("--accountant", default="rdp", choices=["rdp", "gdp", "prv"])
    parser.add_argument("--sigma", type=_positive_float, default=None)
    parser.add_argument("--delta", type=_positive_float, default=None)
    parser.add_argument(
        "--epsilon-mode",
        default=None,
        choices=["none", "calibrate", "hard-stop"],
        help=(
            "calibrate fixes the full training horizon then solves for sigma; hard-stop uses "
            "fixed sigma and checks epsilon after every logical DP step."
        ),
    )
    parser.add_argument("--target-epsilon", type=_positive_float, default=None)
    parser.add_argument(
        "--epsilon-tolerance",
        type=_positive_float,
        default=None,
        help=(
            "Numerical tolerance for Opacus noise calibration. Main-paper presets fix "
            "1e-5, which reproduces all three-decimal sigma values in Table 2."
        ),
    )
    parser.add_argument(
        "--secure-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use Opacus cryptographically secure randomness. Camera-ready numerical runs "
            "used false; real sensitive-data deployment should use true."
        ),
    )
    parser.add_argument(
        "--grad-sample-mode",
        default="hooks",
        choices=["hooks", "ghost", "ghost_fsdp", "ew"],
    )
    parser.add_argument("--C0", type=_positive_float, default=None)

    parser.add_argument(
        "--K",
        type=_positive_int,
        default=None,
        help=(
            "Explicit slack dimension (including paper ablations). If omitted, choose the "
            "largest integer below Eq. (14) using final sigma and expected logical batch."
        ),
    )
    parser.add_argument("--eta", type=_positive_float, default=None)
    parser.add_argument("--beta", type=_fraction_001_099, default=0.5)
    parser.add_argument(
        "--gamma",
        type=_fraction_001_099,
        default=0.5,
        help="Target unclipped fraction for SlaClip-Q and Adap-Clip.",
    )
    parser.add_argument("--c-min", type=_positive_float, default=0.1)
    parser.add_argument("--c-max", type=_positive_float, default=20.0)
    parser.add_argument("--slot-fb-beta", type=_fraction_001_099, default=None)

    parser.add_argument(
        "--strict-paper-check", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--dc-percentile", type=_fraction_001_099, default=0.3)
    parser.add_argument("--dc-stride", type=_positive_float, default=1.0)
    parser.add_argument("--dc-bin-count", type=_positive_int, default=20)
    parser.add_argument("--dc-histogram-std", type=_positive_float, default=5.0)

    # Explicitly retained only to produce an actionable error instead of silently
    # reproducing the old ambiguous behaviour.
    parser.add_argument(
        "--calibrate-sigma", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--use-paper-budgets", action="store_true", help=argparse.SUPPRESS
    )

    parser.add_argument("--max-sequence-length", type=_positive_int, default=256)
    parser.add_argument("--embedding-dim", type=_positive_int, default=128)
    parser.add_argument("--hidden-size", type=_positive_int, default=128)
    parser.add_argument("--n-layers", type=_positive_int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rnn-arch", default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--run-name", default="slaclip_run")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing run's CSV/JSON/config files.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    legacy_flags = {
        "--deficlip-update-mode",
        "--deficlip",
        "--dp-method",
        "--num-slots",
        "-c",
        "--arch",
        "--dp",
    }
    for flag in legacy_flags:
        if flag in raw_argv:
            raise SystemExit(
                f"Legacy flag '{flag}' is not supported. Use paper naming only."
            )

    # parse_args (not parse_known_args) is intentional: misspelled experimental
    # arguments must fail loudly rather than silently changing a run.
    return build_parser().parse_args(raw_argv)
