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

import logging
from collections import defaultdict
from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


logger = logging.getLogger(__name__)
logger.disabled = True


def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.
    """
    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.
    """
    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously used
    in optimization step.
    """
    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _randn_like_compat(
    reference: Union[torch.Tensor, DTensor],
    *,
    generator=None,
) -> Union[torch.Tensor, DTensor]:
    """
    Compatibility helper:
      - Newer torch supports randn_like(..., generator=...)
      - Older torch does NOT -> fallback:
          * For DTensor: keep DTensor by calling randn_like(reference) without generator
          * For plain Tensor: use torch.randn(shape, device, dtype, generator=...)
    """
    # Try modern API first
    if generator is not None:
        try:
            return torch.randn_like(reference, generator=generator)
        except TypeError:
            # older torch, fall through to compatibility path
            pass

    # If generator is None or unsupported:
    if isinstance(reference, DTensor):
        # Keep DTensor type; cannot safely reconstruct DTensor via torch.randn(shape,...)
        return torch.randn_like(reference)

    # Plain Tensor fallback: use torch.randn which has supported generator for a long time
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _generate_noise(
    std: float,
    reference: Union[torch.Tensor, DTensor],
    generator=None,
    secure_mode: bool = False,
) -> Union[torch.Tensor, DTensor]:
    """
    Generates Gaussian noise with mean 0 and std `std`, matching `reference` in
    shape/device/dtype (and DTensor-ness if applicable).

    If secure_mode=True, use Opacus-style defense against floating point attacks:
      - discard first Gaussian sample
      - sum 4 i.i.d samples and divide by 2 (keeps variance == std^2)
    """
    std = float(std)
    zeros = torch.zeros_like(reference)
    if std == 0.0:
        return zeros

    if secure_mode:
        # discard one sample
        _ = _randn_like_compat(reference, generator=generator)

        acc = zeros
        for _i in range(4):
            acc = acc + _randn_like_compat(reference, generator=generator) * std
        return acc / 2.0

    return _randn_like_compat(reference, generator=generator) * std


class DPOptimizer(Optimizer):
    """
    torch.optim.Optimizer wrapper that clips per-sample gradients and adds Gaussian noise.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean"
            )

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        for p in self.params:
            p.summed_grad = None

    def _get_flat_grad_sample(self, p: torch.Tensor):
        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per sample gradient not found. Are you using GradSampleModule?"
            )
        if p.grad_sample is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")
        return ret

    def signal_skip_step(self, do_skip=True):
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self, pop_next=True):
        if self._step_skip_queue:
            if pop_next:
                return self._step_skip_queue.pop(0)
            else:
                return self._step_skip_queue[0]
        else:
            return False

    @property
    def params(self) -> List[nn.Parameter]:
        return params(self)

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        ret = []
        for p in self.params:
            ret.append(self._get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        vals = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )
            if isinstance(p.grad_sample, torch.Tensor):
                vals.append(1)
            elif isinstance(p.grad_sample, list):
                vals.append(len(p.grad_sample))
            else:
                raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        if len(set(vals)) > 1:
            raise ValueError(
                "Number of accumulated steps is inconsistent across parameters"
            )
        return vals[0]

    @property
    def param_groups(self) -> List[dict]:
        return self.original_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups: List[dict]):
        self.original_optimizer.param_groups = param_groups

    @property
    def state(self) -> defaultdict:
        return self.original_optimizer.state

    @state.setter
    def state(self, state: defaultdict):
        self.original_optimizer.state = state

    @property
    def defaults(self) -> dict:
        return self.original_optimizer.defaults

    @defaults.setter
    def defaults(self, defaults: dict):
        self.original_optimizer.defaults = defaults

    def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
        self.step_hook = fn

    def clip_and_accumulate(self):
        grad_samples = self.grad_samples
        if grad_samples is None or len(grad_samples) == 0:
            return

        if len(grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros((0,), device=grad_samples[0].device)
        else:
            per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in grad_samples]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            grad_sample = grad_sample.to(p.dtype)
            clip_factor = per_sample_clip_factor.to(dtype=p.dtype, device=grad_sample.device)

            grad = torch.einsum("i,i...", clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

    def scale_grad(self):
        if self.loss_reduction == "mean":
            denom = self.expected_batch_size * self.accumulated_iterations
            for p in self.params:
                p.grad /= denom

    def zero_grad(self, set_to_none: bool = False):
        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            p.grad_sample = None
            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return True

        self.clip_and_accumulate()

        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return None

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)
