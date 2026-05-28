#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PATCHES_DIR = _THIS_DIR / "patches"
sys.path.insert(0, str(_PATCHES_DIR))
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"[verify] repo_root = {repo_root}")

    try:
        import opacus
        import opacus.privacy_engine
        from opacus import PrivacyEngine 
    except Exception as e:
        print(f"FAIL: unable to import opacus or PrivacyEngine: {e}")
        print("Fix: run from the Opacus repo root, install deps, then `pip install -e .`.")
        return 1

    opacus_path = Path(getattr(opacus, "__file__", "")).resolve() if getattr(opacus, "__file__", None) else None
    pe_path = Path(opacus.privacy_engine.__file__).resolve()
    print(f"opacus.__file__ = {opacus_path}")
    print(f"opacus.privacy_engine = {pe_path}")

    if opacus_path and ("site-packages" in str(opacus_path) or "dist-packages" in str(opacus_path)):
        print("FAIL: opacus resolved to site-packages.")
        print("Fix: run `pip install -e .` in the Opacus repo root and avoid PYTHONPATH shadowing.")
        return 1

    if opacus_path is not None:
        try:
            if not opacus_path.is_relative_to(repo_root):
                print("FAIL: opacus is not resolved from the local repo root.")
                print(f"Resolved: {opacus_path}")
                print("Fix: run from repo root and re-run `pip install -e .`.")
                return 1
        except AttributeError:
            if not str(opacus_path).startswith(str(repo_root)):
                print("FAIL: opacus is not resolved from the local repo root.")
                print(f"Resolved: {opacus_path}")
                print("Fix: run from repo root and re-run `pip install -e .`.")
                return 1

    patches_dir = Path(__file__).resolve().parent / "patches"
    try:
        if not pe_path.is_relative_to(patches_dir):
            print("FAIL: privacy_engine is not resolved from SlaClip patches.")
            print(f"Resolved: {pe_path}")
            print("Fix: ensure SlaClip/patches is inserted before importing opacus.")
            return 1
    except AttributeError:
        if not str(pe_path).startswith(str(patches_dir)):
            print("FAIL: privacy_engine is not resolved from SlaClip patches.")
            print(f"Resolved: {pe_path}")
            print("Fix: ensure SlaClip/patches is inserted before importing opacus.")
            return 1

    pkg_paths = list(getattr(opacus, "__path__", []))
    print(f"opacus.__path__ = {pkg_paths}")
    patches_pkg = patches_dir / "opacus"
    if str(patches_pkg) not in [str(p) for p in pkg_paths]:
        print("FAIL: opacus.__path__ does not include SlaClip patch package path.")
        return 1
    upstream_pkg = repo_root / "opacus"
    if str(upstream_pkg) not in [str(p) for p in pkg_paths]:
        print("FAIL: opacus.__path__ does not include upstream opacus package path.")
        return 1

    try:
        import opacus.optimizers.utils 
    except Exception as e:
        print(f"FAIL: unable to import opacus.optimizers.utils: {e}")
        return 1

    opt_paths = list(getattr(opacus.optimizers, "__path__", []))
    print(f"opacus.optimizers.__path__ = {opt_paths}")
    if str(patches_dir / "opacus" / "optimizers") not in [str(p) for p in opt_paths]:
        print("FAIL: opacus.optimizers.__path__ missing patch path")
        return 1
    if str(upstream_pkg / "optimizers") not in [str(p) for p in opt_paths]:
        print("FAIL: opacus.optimizers.__path__ missing upstream path")
        return 1

    print("OK: opacus resolved from local repo; privacy_engine and optimizers overlay are active.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
