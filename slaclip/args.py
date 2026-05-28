from __future__ import annotations

import argparse
import math
import sys


_PAPER_RECOMMENDED_K_SIGMA1 = {
    128: 8,
    256: 10,
    512: 20,
    1024: 30,
    2048: 50,
}
_Z_0995 = 2.576


def paper_recommended_k(batch_size: int, sigma: float = 1.0) -> int:
    batch_size = int(batch_size)
    sigma = float(sigma)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if math.isclose(sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        exact = _PAPER_RECOMMENDED_K_SIGMA1.get(batch_size)
        if exact is not None:
            return int(exact)

    if sigma <= 0:
        raise ValueError(
            "Automatic K selection for non-tabulated batch sizes requires sigma > 0."
        )

    k_max = (batch_size / (2.0 * _Z_0995 * sigma)) ** (2.0 / 3.0)
    return max(1, int(math.floor(k_max)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SlaClip paper reproduction runner")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["slaclip", "slaclip-q", "vanilla-clip", "adap-clip", "dc-sgd-e", "autoclip", "nondp"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fmnist", "cifar10", "imdb", "names"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-test", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--optim", type=str, default="SGD", choices=["SGD", "Adam", "RMSprop"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-schedule", type=str, default=None, choices=["constant", "cos"])

    parser.add_argument("--accountant", type=str, default="rdp", choices=["rdp", "gdp", "prv"])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--grad-sample-mode", type=str, default="hooks", choices=["hooks", "ghost", "ghost_fsdp", "ew"])
    parser.add_argument("--C0", type=float, default=1.0, help="Initial clipping threshold C0")

    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Slack dimension K. If omitted, auto-select from the paper rule.",
    )
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5, help="Target clipping fraction for slaclip-q and adap-clip")
    parser.add_argument("--c-min", type=float, default=0.1)
    parser.add_argument("--c-max", type=float, default=50.0)
    parser.add_argument("--slot-fb-beta", type=float, default=None)

    parser.add_argument("--strict-paper-check", action="store_true", default=True)
    parser.add_argument("--no-strict-paper-check", action="store_false", dest="strict_paper_check")

    parser.add_argument("--target-epsilon", type=float, default=None)
    parser.add_argument(
        "--calibrate-sigma",
        action="store_true",
        default=False,
        help="Calibrate sigma from target epsilon, delta, epochs, and sample rate.",
    )
    parser.add_argument(
        "--use-paper-budgets",
        action="store_true",
        default=False,
        help="Enable Table-1 epsilon budgets and stop when reached.",
    )

    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rnn-arch", type=str, default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--run-name", type=str, default="slaclip_run")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--data-root", type=str, default="data")

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args, _unknown = parser.parse_known_args()

    legacy_flags = {"--deficlip-update-mode", "--deficlip", "--dp-method", "--num-slots", "-c", "--arch", "--dp"}
    for lf in legacy_flags:
        if lf in sys.argv:
            raise SystemExit(f"Legacy flag '{lf}' is not supported. Use paper naming only.")
    return args
