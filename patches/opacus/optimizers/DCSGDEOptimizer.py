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
import math
from typing import Callable, Optional

import torch
from opt_einsum import contract
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)

logger = logging.getLogger(__name__)

_MIN_STRIDE_ABS = 1e-12
_MIN_STRIDE_REL = 1e-6  # relative to current clipping norm
_EXPERIMENT_C_MIN = 0.1
_EXPERIMENT_C_MAX = 20.0
_PERCENTILE_MIN = 0.01
_PERCENTILE_MAX = 0.99
_DEFAULT_MAX_BOUNDARY_SEARCH_ITERATIONS = 32
_STATE_DICT_KEY = "_slaclip_dcsgde_state"


class DCSGDEOptimizer(DPOptimizer):
    """
    DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation
    Paper: https://arxiv.org/abs/2503.22988
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,  # sigma_total (what user passes / what accountant must see)
        histogram_std: float = 6.0,  # sigma_hist
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        batchsize_train: int = 256,
        dimension: int = 11181642,
        percentile: float = 0.3,  # kept for API compatibility
        stride: float = 1.0,
        bin_cnt: int = 20,
        c_min: Optional[float] = None,
        c_max: Optional[float] = None,
        max_boundary_search_iterations: int = _DEFAULT_MAX_BOUNDARY_SEARCH_ITERATIONS,
        **kwargs,
    ):
        # --- validate ---
        sigma_total = float(noise_multiplier)
        sigma_hist = float(histogram_std)
        if sigma_total < 0:
            raise ValueError("noise_multiplier must be >= 0.")
        if sigma_hist <= 0:
            raise ValueError("histogram_std must be > 0.")
        if not (_EXPERIMENT_C_MIN <= float(max_grad_norm) <= _EXPERIMENT_C_MAX):
            raise ValueError("max_grad_norm must be in [0.1, 20.0].")
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected loss_reduction: {loss_reduction}")

        # --- derive sigma_grad (what we will use to noise gradients) ---
        # 1/sigma_total^2 = 1/sigma_grad^2 + 1/sigma_hist^2
        # => 1/sigma_grad^2 = 1/sigma_total^2 - 1/sigma_hist^2
        if sigma_total == 0.0:
            sigma_grad = 0.0
        else:
            inv_total = sigma_total ** (-2)
            inv_hist = sigma_hist ** (-2)
            inv_grad = inv_total - inv_hist
            if inv_grad <= 0:
                raise ValueError(
                    "Invalid DCSGD-E config: derived inv_sigma_grad_sq <= 0. "
                    "You must have noise_multiplier (sigma_total) < histogram_std (sigma_hist). "
                    f"Got sigma_total={sigma_total}, sigma_hist={sigma_hist}."
                )
            sigma_grad = inv_grad ** (-0.5)

        # Save both:
        self._accountant_noise_multiplier = sigma_total  # what accountant must see
        self._grad_noise_multiplier = float(sigma_grad)  # what we use in add_noise
        self.historgram_std = sigma_hist  # keep original typo name used by old code
        self.histogram_std = sigma_hist  # and a correct alias

        # clip bounds (optional)
        self.c_min = float(c_min) if c_min is not None else _EXPERIMENT_C_MIN
        self.c_max = float(c_max) if c_max is not None else _EXPERIMENT_C_MAX
        if not (_EXPERIMENT_C_MIN <= self.c_min <= _EXPERIMENT_C_MAX):
            raise ValueError("c_min must be in [0.1, 20.0] for DCSGDE.")
        if not (_EXPERIMENT_C_MIN <= self.c_max <= _EXPERIMENT_C_MAX):
            raise ValueError("c_max must be in [0.1, 20.0] for DCSGDE.")
        if self.c_max < self.c_min:
            raise ValueError(
                f"c_max ({self.c_max}) must be >= c_min ({self.c_min}) for DCSGDE"
            )
        if not (self.c_min <= float(max_grad_norm) <= self.c_max):
            raise ValueError("max_grad_norm must lie inside [c_min, c_max] for DCSGDE.")

        # Initialize DPOptimizer with sigma_total so step_hook reads correct sigma
        super().__init__(
            optimizer,
            noise_multiplier=self._accountant_noise_multiplier,
            max_grad_norm=float(max_grad_norm),
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        # --- config / state (kept close to original author code) ---
        # If dimension is not provided or invalid, infer from parameter count.
        if int(dimension) <= 0:
            try:
                dimension = sum(int(p.numel()) for p in self.params)
            except Exception:
                dimension = 0

        # Prefer a sensible default if user didn't pass batchsize_train
        if int(batchsize_train) <= 0:
            batchsize_train = (
                int(expected_batch_size) if expected_batch_size is not None else 1
            )

        self.batchsize_train = int(batchsize_train)
        self.dimension = int(dimension)
        self.percentile = float(percentile)
        if not (_PERCENTILE_MIN <= self.percentile <= _PERCENTILE_MAX):
            raise ValueError("percentile must be in [0.01, 0.99].")

        self.timer = 0
        self.stride = float(stride)
        self.bin_cnt = int(bin_cnt)
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")
        if self.bin_cnt <= 0:
            raise ValueError("bin_cnt must be a positive integer.")
        self.max_boundary_search_iterations = int(max_boundary_search_iterations)
        if self.max_boundary_search_iterations <= 0:
            raise ValueError("max_boundary_search_iterations must be positive.")
        self._last_boundary_search_iterations = 0

        self.stride = max(self.stride, self._get_stride_floor())
        self.max_grad_norm = self._clamp_C(float(self.max_grad_norm))

        # state
        self.sample_size = 0
        self.unclipped_num = 0
        self.norm_stack: list[float] = []
        self._histogram_query_pending = False

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[DCSGDE] sigma_total(accountant)=%.6f, sigma_grad(grad_noise)=%.6f, sigma_hist=%.6f, "
                "C0=%.6f, dim=%d, batchsize_train=%d, stride=%.4f, bins=%d, c_min=%.6f, c_max=%s",
                self._accountant_noise_multiplier,
                self._grad_noise_multiplier,
                self.histogram_std,
                float(self.max_grad_norm),
                int(self.dimension),
                int(self.batchsize_train),
                float(self.stride),
                int(self.bin_cnt),
                float(self.c_min),
                "inf" if self.c_max == float("inf") else f"{self.c_max:.6f}",
            )

    # -------------------------
    # Helpers
    # -------------------------
    def _scalar_hist_noise(self, device: torch.device) -> float:
        """
        Sample 1 scalar Gaussian noise with std = histogram_std.
        Uses Opacus secure noise path if secure_mode=True and respects generator.
        """
        ref = torch.zeros((), device=device, dtype=torch.float32)
        n = _generate_noise(
            std=float(self.histogram_std),
            reference=ref,
            generator=self.generator,
            secure_mode=self.secure_mode,
        )
        return float(n.item())

    # -------------------------
    # Original logic (with safety guards + correct sigma usage)
    # -------------------------
    def before_clip(self):
        """
        Update max_grad_norm based on a privatized histogram of per-sample gradient norms
        collected during clip_and_accumulate() for the previous step.

        NOTE: Variance term uses sigma_grad (not sigma_total), matching author code intent.
        """
        min_stride = self._get_stride_floor()
        stride = max(float(self.stride), min_stride)
        bin_cnt = int(self.bin_cnt)
        self.timer += 1

        # Build the query even for an empty Poisson batch. Branching on the raw
        # realized batch size would expose a private sampling event through the
        # threshold trajectory. A zero histogram plus Gaussian noise follows the
        # same accounted mechanism and makes the subsequent branch DP
        # post-processing.
        hist = [0.0 for _ in range(bin_cnt)]
        for tmp in self.norm_stack:
            if not math.isfinite(tmp):
                continue
            idx = int(tmp / stride)
            if idx > bin_cnt - 1:
                hist[bin_cnt - 1] += 1.0
            elif idx < 0:
                hist[0] += 1.0
            else:
                hist[idx] += 1.0

        # Add Gaussian noise to each bin (DP histogram)
        # (use Opacus generator/secure_mode)
        noise_sum = 0.0
        for i in range(bin_cnt):
            hist[i] += self._scalar_hist_noise(device=torch.device("cpu"))
            noise_sum += hist[i]

        # Numerical guard: if noise_sum is degenerate, skip update
        if not (noise_sum > 0.0):
            self.norm_stack = []
            self._histogram_query_pending = False
            return

        # Search best_cb among {0.1C, 0.2C, ..., 2.0C} (same as author code).
        # The paper/author implementation expands the search again when the minimizer
        # is on a boundary. A hard experiment range makes an unbounded while-loop
        # unsafe, so stop once the bound prevents further expansion, or after the
        # explicit iteration cap and keep the best candidate from the last search.
        best_cb = float(self.max_grad_norm)
        self._last_boundary_search_iterations = 0
        for search_iteration in range(1, self.max_boundary_search_iterations + 1):
            self._last_boundary_search_iterations = search_iteration
            mins = float("inf")
            C_cur = float(self.max_grad_norm)

            for i in range(1, 21):
                cb = (C_cur / 10.0) * float(i)

                # variance term uses sigma_grad (the actual gradient noise)
                # var = sigma_grad^2 * cb^2 * d / (B^2)
                sigma_g = float(self._grad_noise_multiplier)
                var = (sigma_g * sigma_g * cb * cb * float(self.dimension)) / (
                    float(self.batchsize_train) * float(self.batchsize_train)
                )

                bias = 0.0
                for j in range(bin_cnt):
                    mid = (stride / 2.0) + stride * float(j)
                    if mid > cb:
                        bias += hist[j] * (mid - cb) * (mid - cb)

                bias /= noise_sum
                expect = bias + var
                if expect < mins:
                    mins = expect
                    best_cb = cb

            lower_boundary = 0.1 * C_cur
            upper_boundary = 2.0 * C_cur
            at_boundary = math.isclose(best_cb, lower_boundary) or math.isclose(
                best_cb, upper_boundary
            )
            next_c = self._clamp_C(float(best_cb))
            self.max_grad_norm = next_c

            if not at_boundary:
                break

            # The requested expansion cannot move beyond the configured C range.
            if math.isclose(next_c, C_cur):
                break
        else:
            logger.warning(
                "[DCSGDE] boundary search reached %d iterations; "
                "using the final paper-grid candidate C=%.6f.",
                self.max_boundary_search_iterations,
                float(self.max_grad_norm),
            )

        # Update stride heuristics (author code)
        if hist[bin_cnt - 1] > (noise_sum * 0.5):
            self.stride = max(stride * 2.0, min_stride)
        else:
            hist_sum = 0.0
            for i in range(int(0.5 * bin_cnt), bin_cnt):
                hist_sum += hist[i]
            if hist_sum < (noise_sum / float(bin_cnt)):
                new_stride = stride / 2.0
                self.stride = max(new_stride, min_stride)

        self.norm_stack = []
        self._histogram_query_pending = False

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients and per-step buffers.
        """
        super().zero_grad(set_to_none)
        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        """
        Standard DP-SGD clipping (Abadi-style) + collect per-sample norms for histogram update.
        """
        grad_samples = self.grad_samples
        if not grad_samples:
            return
        if len(grad_samples[0]) == 0:
            per_sample_norms = grad_samples[0].new_zeros((0,))
        else:
            per_param_norms = [
                grad_sample.reshape(len(grad_sample), -1).norm(2, dim=-1)
                for grad_sample in grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        # Preserve this flag across physical BatchMemoryManager chunks and clear
        # it only after the one noisy histogram release for the logical step.
        self._histogram_query_pending = True

        # Store norms as python floats (safe across CPU/CUDA and avoids device issues later)
        # This is slower but faithful to the author-style estimator.
        for n in per_sample_norms.detach():
            self.norm_stack.append(float(n.item()))

        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Add noise to gradients using sigma_grad (derived), while keeping sigma_total visible
        to accountant hook (which runs after add_noise() inside DPOptimizer.pre_step()).
        """
        orig_nm = float(self.noise_multiplier)  # should be sigma_total
        try:
            if float(self._grad_noise_multiplier) != orig_nm:
                self.noise_multiplier = float(self._grad_noise_multiplier)
            super().add_noise()
        finally:
            # restore so accountant step_hook sees sigma_total
            self.noise_multiplier = orig_nm

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Run DP step then update clipping threshold for next step.
        """
        pre_step_full = super().pre_step(closure)
        if pre_step_full:
            self.before_clip()
        return pre_step_full

    def _clamp_C(self, C_val: float) -> float:
        """
        Enforce [c_min, c_max] bounds (if provided) plus absolute minimum.
        """
        return min(max(C_val, self.c_min), self.c_max)

    def _get_stride_floor(self) -> float:
        """
        Compute smallest allowed stride based on current clipping norm.
        Keeps the histogram sensitive for tiny-gradient datasets while
        avoiding division-by-zero.
        """
        C_cur = float(self.max_grad_norm)
        rel_floor = abs(C_cur) * _MIN_STRIDE_REL
        return max(_MIN_STRIDE_ABS, rel_floor)

    def state_dict(self):
        """Checkpoint only DP-post-processed state at a logical-step boundary."""
        if (
            self._is_last_step_skipped
            or self.norm_stack
            or self._histogram_query_pending
            or self._has_unprocessed_private_state()
        ):
            raise RuntimeError(
                "Cannot checkpoint DCSGDE during an incomplete logical batch: "
                "norm_stack contains raw per-sample gradient norms. Checkpoint only "
                "after a complete optimizer step."
            )

        state = super().state_dict()
        state[_STATE_DICT_KEY] = {
            "version": 1,
            # These values are public counters or post-processing of a DP histogram.
            "max_grad_norm": float(self.max_grad_norm),
            "timer": int(self.timer),
            "stride": float(self.stride),
            "last_boundary_search_iterations": int(
                self._last_boundary_search_iterations
            ),
        }
        return state

    def _has_unprocessed_private_state(self) -> bool:
        """Return whether backward/clipping has not completed its noisy release."""
        for parameter in self.params:
            for attribute in ("grad_sample", "summed_grad"):
                value = getattr(parameter, attribute, None)
                tensors = value if isinstance(value, list) else [value]
                if any(
                    isinstance(tensor, torch.Tensor)
                    and not hasattr(tensor, "_processed")
                    for tensor in tensors
                ):
                    return True
        return False

    def load_state_dict(self, state_dict) -> None:
        """Restore both the wrapped optimizer and histogram-controller state."""
        controller_state = state_dict.get(_STATE_DICT_KEY)
        optimizer_state = {
            key: value for key, value in state_dict.items() if key != _STATE_DICT_KEY
        }
        super().load_state_dict(optimizer_state)

        # Backwards compatibility with checkpoints produced before controller state
        # was included.
        if controller_state is None:
            return

        restored_c = float(controller_state["max_grad_norm"])
        if not (math.isfinite(restored_c) and self.c_min <= restored_c <= self.c_max):
            raise ValueError(
                "DCSGDE checkpoint max_grad_norm lies outside the configured bounds."
            )

        restored_stride = float(controller_state["stride"])
        if not (math.isfinite(restored_stride) and restored_stride > 0):
            raise ValueError("DCSGDE checkpoint stride must be finite and positive.")

        self.max_grad_norm = restored_c
        self.timer = int(controller_state.get("timer", 0))
        self.stride = max(restored_stride, self._get_stride_floor())
        # Raw/in-flight per-sample data is intentionally never serialized.
        self.sample_size = 0
        self.unclipped_num = 0
        self.norm_stack = []
        self._histogram_query_pending = False
        self._last_boundary_search_iterations = int(
            controller_state.get("last_boundary_search_iterations", 0)
        )
