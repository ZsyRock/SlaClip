from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch


def ensure_output_dir(out_dir: str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_output_paths_available(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite an existing run. Choose a unique --run-name or pass "
            f"--overwrite explicitly. Existing: {existing}"
        )


def _atomic_text_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _git_snapshot(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), *arguments],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = run("status", "--porcelain=v1")
    return {
        "path": str(repo.resolve()),
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "describe": run("describe", "--always", "--dirty", "--tags") or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or ""
        if name:
            packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def collect_run_metadata(
    *,
    args,
    protocol_metadata: dict,
    data_metadata: dict,
    opacus_root: Path,
    slaclip_root: Path,
    device: torch.device,
    logical_steps_per_epoch: int,
    expected_batch_size: int,
) -> dict[str, Any]:
    cuda_device = None
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        cuda_device = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(device)),
        }

    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "configuration": dict(sorted(vars(args).items())),
        "protocol": protocol_metadata,
        "privacy_scope": {
            "adjacency": "add/remove (unbounded)",
            "sampling": "Poisson",
            "accountant": str(args.accountant),
            "delta": float(args.delta),
            "private_training_metrics_published": False,
            "training_metric_note": (
                "For private methods, raw training loss and accuracy are suppressed "
                "because they are not outputs of the accounted Gaussian mechanism."
            ),
            "selection_note": (
                "For phase=selection, the DP guarantee covers private_train_size only; "
                "validation is an explicitly acknowledged public non-DP holdout. This is a "
                "per-command accountant and does not compose a multi-candidate sweep."
            ),
        },
        "data": data_metadata,
        "sampling": {
            "requested_logical_batch_size": int(args.batch_size),
            "max_physical_batch_size": int(args.max_physical_batch_size),
            "logical_steps_per_epoch": int(logical_steps_per_epoch),
            "effective_sample_rate": 1.0 / float(logical_steps_per_epoch),
            "expected_batch_size": int(expected_batch_size),
        },
        "git": {
            "opacus": _git_snapshot(opacus_root),
            "slaclip": _git_snapshot(slaclip_root),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "cuda_device": cuda_device,
            "secure_mode": bool(args.secure_mode),
        },
        "installed_packages": _installed_packages(),
    }


def write_epoch_csv(path: Path, records: List[Dict]) -> None:
    if not records:
        return
    preferred = [
        "epoch",
        "logical_steps_completed",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "test_loss",
        "test_accuracy",
        "epsilon",
        "delta",
        "C_t",
        "dataset",
        "method",
        "protocol",
        "phase",
        "seed",
    ]
    all_fields = {key for record in records for key in record}
    fieldnames = [field for field in preferred if field in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))

    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key, "") for key in fieldnames})
    _atomic_text_write(path, buffer.getvalue())


def write_epoch_json(path: Path, records: List[Dict]) -> None:
    """Backward-compatible epoch-only writer used by external scripts."""

    _atomic_text_write(
        path,
        json.dumps(
            _json_safe(records), indent=2, default=_json_default, allow_nan=False
        )
        + "\n",
    )


def write_run_json(
    path: Path,
    *,
    metadata: dict,
    records: List[Dict],
    final: dict | None,
) -> None:
    payload = {
        "schema_version": 2,
        "metadata": metadata,
        "epochs": records,
        "final": final,
    }
    _atomic_text_write(
        path,
        json.dumps(
            _json_safe(payload), indent=2, default=_json_default, allow_nan=False
        )
        + "\n",
    )


def write_config_json(path: Path, metadata: dict) -> None:
    _atomic_text_write(
        path,
        json.dumps(
            _json_safe(metadata), indent=2, default=_json_default, allow_nan=False
        )
        + "\n",
    )
