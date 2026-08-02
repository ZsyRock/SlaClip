#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SLACLIP_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_SLACLIP_ROOT))

from tools.main_grid_common import (  # noqa: E402
    read_jsonl,
    read_run_payloads,
    select_best_candidates,
    validate_complete_grid,
    write_commands,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every main-grid selection run, choose by final validation "
            "accuracy only, and generate (without executing) seeds 42/43/44 retrains."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Retrain JSONL")
    parser.add_argument(
        "--commands-output",
        type=Path,
        required=True,
        help="Generated retrain shell list",
    )
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    validate_complete_grid(candidates)
    payloads = read_run_payloads(candidates, args.runs_dir)
    retrains = select_best_candidates(candidates, payloads)
    write_jsonl(args.output, retrains)
    write_commands(args.commands_output, retrains)
    print(
        f"Validated {len(candidates)} candidate runs and generated {len(retrains)} "
        f"retrain entries in {args.output} and {args.commands_output}; no command was executed."
    )


if __name__ == "__main__":
    main()
