"""Make the nested overlay testable from the parent Opacus-Aug directory."""

from __future__ import annotations

import sys
from pathlib import Path


_SLACLIP_ROOT = Path(__file__).resolve().parents[1]
_PATCHES_ROOT = _SLACLIP_ROOT / "patches"
_OPACUS_ROOT = _SLACLIP_ROOT.parent

for _path in reversed((_PATCHES_ROOT, _SLACLIP_ROOT, _OPACUS_ROOT)):
    value = str(_path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
