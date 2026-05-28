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
        if not (0.0 <= target_unclipped_quantile <= 1.0):
            raise ValueError("target_unclipped_quantile must be in [0, 1].")
        if clipbound_learning_rate <= 0:
            raise ValueError("clipbound_learning_rate must be > 0.")
        if max_clipbound <= 0 or min_clipbound <= 0:
            raise ValueError("max_clipbound and min_clipbound must be > 0.")
        if max_clipbound <= min_clipbound:
            raise ValueError("max_clipbound must be larger than min_clipbound.")
        unclipped_num_std = float(unclipped_num_std)
        if unclipped_num_std <= 0:
            raise ValueError("unclipped_num_std must be > 0.")

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
            inv_sigma_total_sq = (self._accountant_noise_multiplier ** -2)
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

        self.sample_size: int = 0
        self.unclipped_num: float = 0.0

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
        Clear gradients and reset counters.
        """
        super().zero_grad(set_to_none)
        self.sample_size = 0
        self.unclipped_num = 0.0

    def clip_and_accumulate(self):
        """
        Clip gradients and update unclipped count.
        """
        per_param_norms = [g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]
        if not per_param_norms:
            return

        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        if per_sample_norms.numel() == 0:
            return

        per_sample_clip_factor = (
            float(self.max_grad_norm) / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)

        bs = int(per_sample_clip_factor.numel())
        self.sample_size += bs
        # unclipped: clip_factor == 1
        self.unclipped_num += float((per_sample_clip_factor >= 1.0).sum().item())

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
        Add noise to gradients and unclipped counts.
        """
        orig_nm = float(self.noise_multiplier)
        try:
            if float(self._grad_noise_multiplier) != orig_nm:
                self.noise_multiplier = float(self._grad_noise_multiplier)
            super().add_noise()
        finally:
            self.noise_multiplier = orig_nm

        if self.sample_size <= 0:
            return

        comm_dev = self._dist_device(torch.device("cpu"))
        t = torch.tensor(
            [float(self.unclipped_num), float(self.sample_size)],
            device=comm_dev,
            dtype=torch.float32,
        )
        self._all_reduce_inplace_sum(t)

        unclipped_total = float(t[0].item())
        sample_size_total = int(round(float(t[1].item())))
        sample_size_total = max(sample_size_total, 0)

        ref = torch.tensor(unclipped_total, device=comm_dev, dtype=torch.float32)
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

        self.unclipped_num = float(unclipped_total + float(noise.item()))
        self.sample_size = int(sample_size_total)

    def update_max_grad_norm(self):
        """
        Update C and clamp to [min_clipbound, max_clipbound].
        """
        if self.sample_size <= 0:
            return

        unclipped_frac = float(self.unclipped_num) / float(self.sample_size)
        unclipped_frac = max(0.0, min(1.0, unclipped_frac))

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

    def pre_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
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
