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
import torch.distributed as dist
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)

logger = logging.getLogger(__name__)

_EXPERIMENT_C_MIN = 0.1
_EXPERIMENT_C_MAX = 20.0
_PERCENTILE_MIN = 0.01
_PERCENTILE_MAX = 0.99
_STATE_DICT_KEY = "_slaclip_adaclip_state"


class AdaClipDPOptimizer(DPOptimizer):
    """
    AdaClip: Differentially Private Learning with Adaptive Clipping
    Paper: https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        target_unclipped_quantile: float,
        clipbound_learning_rate: float,
        max_clipbound: float,
        min_clipbound: float,
        unclipped_num_std: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        noise_multiplier = float(noise_multiplier)
        if not (_PERCENTILE_MIN <= float(target_unclipped_quantile) <= _PERCENTILE_MAX):
            raise ValueError("target_unclipped_quantile must be in [0.01, 0.99].")
        if clipbound_learning_rate <= 0:
            raise ValueError("clipbound_learning_rate must be > 0.")
        if not (_EXPERIMENT_C_MIN <= float(min_clipbound) <= _EXPERIMENT_C_MAX):
            raise ValueError("min_clipbound must be in [0.1, 20.0].")
        if not (_EXPERIMENT_C_MIN <= float(max_clipbound) <= _EXPERIMENT_C_MAX):
            raise ValueError("max_clipbound must be in [0.1, 20.0].")
        if max_clipbound <= min_clipbound:
            raise ValueError("max_clipbound must be larger than min_clipbound.")
        if not (float(min_clipbound) <= float(max_grad_norm) <= float(max_clipbound)):
            raise ValueError(
                "max_grad_norm must lie inside [min_clipbound, max_clipbound]."
            )
        unclipped_num_std = float(unclipped_num_std)
        if unclipped_num_std <= 0:
            raise ValueError("unclipped_num_std must be > 0.")
        if loss_reduction != "mean":
            raise ValueError(
                "AdaClip requires loss_reduction='mean' so its privatized "
                "centered count uses a fixed expected-batch denominator."
            )
        if expected_batch_size is None or int(expected_batch_size) <= 0:
            raise ValueError("expected_batch_size must be a positive integer.")

        self._accountant_noise_multiplier = noise_multiplier

        if noise_multiplier > 0 and noise_multiplier >= 2.0 * unclipped_num_std:
            raise ValueError(
                "noise_multiplier must be smaller than 2 * unclipped_num_std. "
                "This is a requirement stemming from Theorem 1 in "
                "https://arxiv.org/pdf/1905.03871.pdf"
            )

        super().__init__(
            optimizer,
            noise_multiplier=self._accountant_noise_multiplier,
            max_grad_norm=float(max_grad_norm),
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        self.target_unclipped_quantile = float(target_unclipped_quantile)
        self.clipbound_learning_rate = float(clipbound_learning_rate)
        self.max_clipbound = float(max_clipbound)
        self.min_clipbound = float(min_clipbound)
        self.unclipped_num_std = float(unclipped_num_std)

        if self._accountant_noise_multiplier > 0:
            inv_sigma_total_sq = self._accountant_noise_multiplier**-2
            inv_2sigb_sq = (2.0 * self.unclipped_num_std) ** -2
            # sigma_grad^{-2} = sigma_total^{-2} - (2*sigma_b)^{-2}
            inv_sigma_grad_sq = inv_sigma_total_sq - inv_2sigb_sq
            if inv_sigma_grad_sq <= 0:
                raise ValueError(
                    "Invalid AdaClip config: derived inv_sigma_grad_sq <= 0. "
                    "Please ensure noise_multiplier < 2 * unclipped_num_std."
                )
            self._grad_noise_multiplier = (inv_sigma_grad_sq) ** (-0.5)
        else:
            self._grad_noise_multiplier = 0.0

        # The paper's Theorem 1 uses the sensitivity-1/2 centered bit
        # (1{||g_i|| <= C} - 1/2).  Keeping that query centered is essential under
        # add/remove Poisson sampling: division by the realized private batch size
        # would make the release distribution data-dependent.  ``sample_size`` is
        # retained only as an in-flight diagnostic and is never released or saved.
        self.sample_size: int = 0
        self.centered_unclipped_sum: float = 0.0
        self.noisy_unclipped_fraction: Optional[float] = None
        # Privacy-sensitive phase flags. Raw state must never be serialized.
        self._count_query_pending = False
        self._controller_update_pending = False

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[AdaClip] sigma_total(accountant)=%.6f, sigma_grad(grad_noise)=%.6f, sigma_b=%.6f, "
                "q=%.3f, lr=%.6f, C0=%.6f",
                self._accountant_noise_multiplier,
                self._grad_noise_multiplier,
                self.unclipped_num_std,
                self.target_unclipped_quantile,
                self.clipbound_learning_rate,
                float(self.max_grad_norm),
            )

    # ----------------- small distributed helpers -----------------
    @staticmethod
    def _dist_device(fallback: torch.device) -> torch.device:
        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_backend() == "nccl":
                    return torch.device("cuda")
            except Exception:
                pass
        return fallback

    @staticmethod
    def _all_reduce_inplace_sum(t: torch.Tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    @staticmethod
    def _broadcast_inplace(t: torch.Tensor, src: int = 0):
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(t, src=src)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients and reset counters after a complete logical batch.

        BatchMemoryManager calls zero_grad() between physical batches while
        ``_is_last_step_skipped`` is true. Preserve the count query across those
        chunks, just as DPOptimizer preserves ``summed_grad``.
        """
        super().zero_grad(set_to_none)
        if not self._is_last_step_skipped:
            self.sample_size = 0
            self.centered_unclipped_sum = 0.0
            self.noisy_unclipped_fraction = None
            self._count_query_pending = False
            self._controller_update_pending = False

    def clip_and_accumulate(self):
        """
        Clip gradients and update unclipped count.
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
        # These flags are cleared only after the complete logical-step controller
        # phase, including an empty Poisson batch. Every logical Poisson step must
        # therefore execute the same Gaussian count-release path.
        self._count_query_pending = True
        self._controller_update_pending = True

        per_sample_clip_factor = (
            float(self.max_grad_norm) / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)

        bs = int(per_sample_clip_factor.numel())
        self.sample_size += bs
        unclipped = per_sample_norms <= float(self.max_grad_norm)
        self.centered_unclipped_sum += float(
            (unclipped.to(dtype=torch.float32) - 0.5).sum().item()
        )

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            clip_factor_on_device = per_sample_clip_factor.to(grad_sample.device)
            grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Add noise to gradients and to the sensitivity-1/2 centered count.

        The released fraction is ``clip(1/2 + (sum_i(b_i-1/2)+Z)/B, 0, 1)``
        with a fixed public expected-batch denominator.  For a fixed-size batch
        this is algebraically identical to the paper's noisy unclipped fraction;
        for Poisson batches it avoids exposing the realized private batch size.
        """
        count_release_denom = float(self.expected_batch_size) * float(
            self.accumulated_iterations
        )
        if dist.is_available() and dist.is_initialized():
            count_release_denom *= float(dist.get_world_size())
        if count_release_denom <= 0:
            raise ValueError("AdaClip count-release denominator must be positive.")

        orig_nm = float(self.noise_multiplier)
        try:
            if float(self._grad_noise_multiplier) != orig_nm:
                self.noise_multiplier = float(self._grad_noise_multiplier)
            super().add_noise()
        finally:
            self.noise_multiplier = orig_nm

        comm_dev = self._dist_device(torch.device("cpu"))
        t = torch.tensor(
            float(self.centered_unclipped_sum),
            device=comm_dev,
            dtype=torch.float32,
        )
        self._all_reduce_inplace_sum(t)

        centered_total = float(t.item())
        ref = torch.tensor(centered_total, device=comm_dev, dtype=torch.float32)
        noise = torch.zeros_like(ref)

        rank = 0
        if dist.is_available() and dist.is_initialized():
            try:
                rank = dist.get_rank()
            except Exception:
                rank = 0

        if rank == 0:
            noise = _generate_noise(
                std=float(self.unclipped_num_std),
                reference=ref,
                generator=self.generator,
                secure_mode=self.secure_mode,
            ).to(device=comm_dev, dtype=torch.float32)

        self._broadcast_inplace(noise, src=0)

        noisy_fraction = (
            0.5 + (centered_total + float(noise.item())) / count_release_denom
        )
        self.noisy_unclipped_fraction = max(0.0, min(1.0, noisy_fraction))

        # Discard all raw query state immediately after the DP release. Only the
        # privatized fraction remains available to the controller.
        self.centered_unclipped_sum = 0.0
        self.sample_size = 0
        self._count_query_pending = False
        self._controller_update_pending = True

    def update_max_grad_norm(self):
        """
        Update C and clamp to [min_clipbound, max_clipbound].
        """
        if self.noisy_unclipped_fraction is None:
            return

        unclipped_frac = float(self.noisy_unclipped_fraction)

        scale = math.exp(
            -float(self.clipbound_learning_rate)
            * (unclipped_frac - float(self.target_unclipped_quantile))
        )
        new_c = float(self.max_grad_norm) * float(scale)
        new_c = max(float(self.min_clipbound), min(float(self.max_clipbound), new_c))

        if dist.is_available() and dist.is_initialized():
            comm_dev = self._dist_device(torch.device("cpu"))
            c_t = torch.tensor(float(new_c), device=comm_dev, dtype=torch.float32)
            self._broadcast_inplace(c_t, src=0)
            new_c = float(c_t.item())

        self.max_grad_norm = float(new_c)
        self.noisy_unclipped_fraction = None
        self._controller_update_pending = False

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Run DP processing before optimizer.step().
        """
        res = super().pre_step(closure)

        should_update = True
        if res is None:
            should_update = False
        elif isinstance(res, bool):
            should_update = bool(res)

        if should_update:
            self.update_max_grad_norm()

        return res

    def state_dict(self):
        """Checkpoint only post-processed state at a complete logical-step boundary."""
        if (
            self._is_last_step_skipped
            or self._count_query_pending
            or self._controller_update_pending
            or self._has_unprocessed_grad_samples()
        ):
            raise RuntimeError(
                "Cannot checkpoint AdaClip during an incomplete logical batch: "
                "the unclipped-count query has not completed its DP release and "
                "controller update. Checkpoint only after a complete optimizer step."
            )

        state = super().state_dict()
        state[_STATE_DICT_KEY] = {
            "version": 1,
            # max_grad_norm is post-processing of a DP count release.
            "max_grad_norm": float(self.max_grad_norm),
        }
        return state

    def _has_unprocessed_grad_samples(self) -> bool:
        """Return whether backward produced gradients not consumed by a DP step."""
        for parameter in self.params:
            grad_sample = getattr(parameter, "grad_sample", None)
            tensors = grad_sample if isinstance(grad_sample, list) else [grad_sample]
            if any(
                isinstance(tensor, torch.Tensor) and not hasattr(tensor, "_processed")
                for tensor in tensors
            ):
                return True
        return False

    def load_state_dict(self, state_dict) -> None:
        """Restore both the wrapped optimizer and adaptive clipping state."""
        controller_state = state_dict.get(_STATE_DICT_KEY)
        optimizer_state = {
            key: value for key, value in state_dict.items() if key != _STATE_DICT_KEY
        }
        super().load_state_dict(optimizer_state)

        # Backwards compatibility: old checkpoints only stored the wrapped optimizer.
        if controller_state is None:
            return

        restored_c = float(controller_state["max_grad_norm"])
        if not (
            math.isfinite(restored_c)
            and self.min_clipbound <= restored_c <= self.max_clipbound
        ):
            raise ValueError(
                "AdaClip checkpoint max_grad_norm lies outside the configured bounds."
            )

        self.max_grad_norm = restored_c
        # Raw/in-flight query buffers are intentionally never serialized.
        self.sample_size = 0
        self.centered_unclipped_sum = 0.0
        self.noisy_unclipped_fraction = None
        self._count_query_pending = False
        self._controller_update_pending = False
