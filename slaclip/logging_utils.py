from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def ensure_output_dir(out_dir: str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_epoch_csv(path: Path, records: List[Dict]) -> None:
    if not records:
        return
    fieldnames = [
        "epoch",
        "epsilon",
        "test_accuracy",
        "C_t",
        "dataset",
        "method",
        "seed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_epoch_json(path: Path, records: List[Dict]) -> None:
    with path.open("w") as f:
        json.dump(records, f, indent=2)
