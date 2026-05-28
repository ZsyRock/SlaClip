# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from typing import Optional, Type

from .adaclipoptimizer import AdaClipDPOptimizer
from .ddp_perlayeroptimizer import SimpleDistributedPerLayerOptimizer
from .ddpoptimizer import DistributedDPOptimizer
from .ddpoptimizer_fast_gradient_clipping import DistributedDPOptimizerFastGradientClipping
from .fsdpoptimizer_fast_gradient_clipping import FSDPOptimizerFastGradientClipping
from .optimizer import DPOptimizer
from .optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from .perlayeroptimizer import DPPerLayerOptimizer

from .slaclipoptimizer import SlaClipOptimizer, SlaClipQOptimizer
from .DCSGDEOptimizer import DCSGDEOptimizer
from .autoclipoptimizer import AutoClipDPOptimizer

__all__ = [
    "DPOptimizer",
    "DistributedDPOptimizer",
    "DPPerLayerOptimizer",
    "SimpleDistributedPerLayerOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientClipping",
    "FSDPOptimizerFastGradientClipping",
    "AdaClipDPOptimizer",
    "AutoClipDPOptimizer",
    "SlaClipOptimizer",
    "SlaClipQOptimizer",
    "DCSGDEOptimizer",
    "get_optimizer_class",
]


def get_optimizer_class(
    clipping: str,
    distributed: bool,
    grad_sample_mode: Optional[str] = None,
) -> Type:
    if grad_sample_mode == "ghost":
        if clipping == "flat" and distributed is False:
            return DPOptimizerFastGradientClipping
        if clipping == "flat" and distributed is True:
            return DistributedDPOptimizerFastGradientClipping
        raise ValueError(
            f"Unsupported combination: clipping={clipping}, distributed={distributed}, grad_sample_mode={grad_sample_mode}"
        )

    if grad_sample_mode == "ghost_fsdp":
        if clipping == "flat" and distributed is True:
            return FSDPOptimizerFastGradientClipping
        raise ValueError(
            f"Unsupported combination: clipping={clipping}, distributed={distributed}, grad_sample_mode={grad_sample_mode}"
        )

    if clipping == "flat" and distributed is False:
        return DPOptimizer
    if clipping == "flat" and distributed is True:
        return DistributedDPOptimizer

    if clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    if clipping == "per_layer" and distributed is True:
        if grad_sample_mode in ("hooks", "ew"):
            return SimpleDistributedPerLayerOptimizer
        raise ValueError(f"Unexpected grad_sample_mode for distributed per-layer: {grad_sample_mode}")

    if clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer

    if clipping == "slaclip" and distributed is False:
        return SlaClipOptimizer

    if clipping == "slaclip-q" and distributed is False:
        return SlaClipQOptimizer

    if clipping == "dc-sgd-e" and distributed is False:
        return DCSGDEOptimizer

    if clipping == "autoclip" and distributed is False:
        return AutoClipDPOptimizer

    raise ValueError(f"Unexpected optimizer parameters: clipping={clipping}, distributed={distributed}")
