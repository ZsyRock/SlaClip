#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SLACLIP_ROOT = _THIS_DIR.parent
_OPACUS_ROOT = _SLACLIP_ROOT.parent
sys.path.insert(0, str(_SLACLIP_ROOT))

from tools.main_grid_common import (  # noqa: E402
    generate_main_candidates,
    source_git_identity,
    write_commands,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, but do not execute, the complete ICML-2026 main-protocol "
            "selection grid. Both source repositories must be clean."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="Candidate JSONL")
    parser.add_argument(
        "--commands-output",
        type=Path,
        required=True,
        help="Generated shell command list",
    )
    args = parser.parse_args()

    source_git = source_git_identity(
        slaclip_root=_SLACLIP_ROOT, opacus_root=_OPACUS_ROOT
    )
    candidates = generate_main_candidates(source_git)
    write_jsonl(args.output, candidates)
    write_commands(args.commands_output, candidates)
    print(
        f"Generated {len(candidates)} selection candidates in {args.output} "
        f"and {args.commands_output}; no command was executed."
    )


if __name__ == "__main__":
    main()
