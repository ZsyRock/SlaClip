"""SlaClip optimizer."""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch.optim import Optimizer

from .optimizer import DPOptimizer, _check_processed_flag, _generate_noise, _mark_as_processed

logger = logging.getLogger(__name__)


class _SlaClipBase(DPOptimizer):

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
        num_slots: int = 10,
        eta: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.5,
        c_min: float = 0.1,
        c_max: float = 50.0,
        strict_paper_check: bool = True,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        self.K = int(num_slots)
        if self.K <= 0:
            raise ValueError("K must be a positive integer")
        self.eta = float(eta)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.strict_paper_check = bool(strict_paper_check)
        self.current_clip = float(max_grad_norm)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.slot_fb_ratio_min = 0.5
        self.slot_fb_ratio_max = 2.0
        self._EPS = 1e-6

        self._sample_count: int = 0
        self._slack_sum: Optional[torch.Tensor] = None
        self._lambda_t: float = 0.0
        self._slack_indicator: Optional[torch.Tensor] = None

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none)
        self._slack_indicator = None
        if not self._is_last_step_skipped:
            self._sample_count = 0
            self._slack_sum = None
            self._lambda_t = 0.0

    def _release_denom(self) -> float:
        if self.loss_reduction == "sum":
            return 1.0

        denom = float(self.expected_batch_size) * float(self.accumulated_iterations)
        if denom <= 0:
            raise ValueError("Expected release denominator must be > 0")
        return denom

    def _build_slack_vector(self, L_ti: torch.Tensor, lambda_t: float) -> torch.Tensor:
        B = int(L_ti.shape[0])
        K = int(self.K)
        slack_vector = torch.zeros(B, K, device=L_ti.device, dtype=torch.float32)
        if K == 0 or lambda_t <= 0:
            return slack_vector

        a = torch.floor(L_ti / lambda_t).to(torch.int64)
        a_clamped = torch.clamp(a, max=K)
        b = L_ti - a_clamped.to(L_ti.dtype) * lambda_t
        b = torch.where(a_clamped >= K, torch.zeros_like(b), b)

        ar = torch.arange(K, device=L_ti.device).view(1, K)
        mask = ar < a_clamped.view(B, 1)
        slack_vector = mask.to(slack_vector.dtype) * float(lambda_t)

        valid = a_clamped < K
        if valid.any():
            idx = torch.clamp(a_clamped, max=K - 1)
            slack_vector[valid, idx[valid]] = b[valid]

        return slack_vector

    def clip_and_accumulate(self):
        grad_samples = self.grad_samples
        if grad_samples is None or len(grad_samples) == 0:
            return

        # Step 1: per-sample norms (g_{t,i})
        B = None
        device = None
        sum_sq = None
        flat_cache = []

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            flat = self._get_flat_grad_sample(p)
            flat = flat.to(dtype=torch.float32)
            flat_cache.append((p, flat))

            if B is None:
                B = int(flat.shape[0])
                device = flat.device
                sum_sq = torch.zeros(B, device=device, dtype=torch.float32)
            else:
                if int(flat.shape[0]) != int(B):
                    raise ValueError("Inconsistent batch dimension across parameters")

            g2 = flat.view(B, -1)
            sum_sq = sum_sq + (g2 * g2).sum(dim=1)

        batch_size = int(B) if B is not None else 0
        if batch_size == 0:
            for p, flat in flat_cache:
                _mark_as_processed(p.grad_sample)
            return

        C_t = float(self.current_clip)
        eps = 1e-12
        norms = torch.sqrt(sum_sq + eps)

        # Clip_{C_t}(g) = g * min(1, C_t / (||g||_2 + 1e-12))
        clip_factor = (C_t / (norms + eps)).clamp(max=1.0)

        for p, flat in flat_cache:
            grad = torch.einsum("i,i...", clip_factor.to(flat.dtype), flat)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad
            _mark_as_processed(p.grad_sample)

        # Eq. (6)-(8): slack encoding into K dims
        slack_amount = torch.clamp(C_t - norms, min=0.0)
        lambda_t = float(C_t / math.sqrt(self.K))
        if self._lambda_t and not math.isclose(
            self._lambda_t, lambda_t, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("Inconsistent lambda_t across accumulated physical batches")
        self._lambda_t = lambda_t
        L_ti = slack_amount * math.sqrt(self.K)
        slack_vector = self._build_slack_vector(L_ti, self._lambda_t)
        batch_slack_sum = slack_vector.sum(dim=0).to(torch.float32)
        if self._slack_sum is None:
            self._slack_sum = batch_slack_sum
        else:
            if self._slack_sum.device != batch_slack_sum.device:
                batch_slack_sum = batch_slack_sum.to(self._slack_sum.device)
            self._slack_sum += batch_slack_sum
        self._sample_count += batch_size

    def _compute_slack_indicator(self, C_t: float) -> Optional[torch.Tensor]:
        if self._slack_sum is None:
            return None
        if self._lambda_t <= 0:
            return None

        noise = _generate_noise(
            std=self.noise_multiplier * C_t,
            reference=self._slack_sum,
            generator=self.generator,
            secure_mode=self.secure_mode,
        )
        slack_noisy_sum = self._slack_sum + noise
        s_hat = slack_noisy_sum / (self._lambda_t * self._release_denom())
        return s_hat

    def _update_threshold(self, C_t: float, s_hat: torch.Tensor) -> float:
        raise NotImplementedError

    def add_noise(self):
        C_t = float(self.current_clip)
        self.max_grad_norm = C_t

        param_slices = []
        total_dim = 0
        first_dev = None
        for p in self.params:
            if p.summed_grad is None:
                continue
            _check_processed_flag(p.summed_grad)
            if first_dev is None:
                first_dev = p.summed_grad.device
            sz = int(p.summed_grad.numel())
            param_slices.append((p, total_dim, sz))
            total_dim += sz

        if first_dev is None:
            return

        noise_ref = torch.empty(
            total_dim + int(self.K), device=first_dev, dtype=torch.float32
        )
        noise_full = _generate_noise(
            std=self.noise_multiplier * C_t,
            reference=noise_ref,
            generator=self.generator,
            secure_mode=self.secure_mode,
        ).to(first_dev, dtype=torch.float32)

        for p, start, sz in param_slices:
            ng = noise_full[start: start + sz].view_as(p.summed_grad)
            p.grad = (p.summed_grad + ng).view_as(p)
            _mark_as_processed(p.summed_grad)

        # Match Opacus DPOptimizer semantics exactly:
        # under mean reduction, the private release is normalized by
        # expected_batch_size * accumulated_iterations, not by the realized
        # Poisson batch size. Using the same denominator keeps the first d
        # coordinates and the K slack coordinates on the same joint release.
        if self._slack_sum is None:
            return
        if self._lambda_t <= 0:
            return

        slack_sum = self._slack_sum
        if slack_sum.device != first_dev:
            slack_sum = slack_sum.to(first_dev)
        slot_noise = noise_full[total_dim: total_dim + int(self.K)]
        slack_noisy_sum = slack_sum + slot_noise
        s_hat = slack_noisy_sum / (self._lambda_t * self._release_denom())
        if s_hat is None:
            return

        self._slack_indicator = s_hat

        C_next = self._update_threshold(C_t, s_hat)
        if math.isfinite(C_next) and C_next > 0:
            self.current_clip = float(C_next)
            self.max_grad_norm = float(C_next)


class SlaClipOptimizer(_SlaClipBase):

    def _update_threshold(self, C_t: float, s_hat: torch.Tensor) -> float:
        # DefiClip slot_feedback update
        if self.strict_paper_check:
            if s_hat.numel() != int(self.K):
                raise ValueError("strict_paper_check: slack_indicator length != K")
        q_hat = float(s_hat[0].item())
        r_hat = float(s_hat[int(self.K) - 1].item())

        z_t = r_hat / (C_t + self._EPS)
        gamma_t = 1.0 - self.beta * (1.0 - z_t)
        gamma_t = float(max(0.0, min(1.0, gamma_t)))

        step = self.eta * (gamma_t - q_hat)
        C_next = float(C_t * math.exp(step))
        return float(max(self.c_min, min(self.c_max, C_next)))


class SlaClipQOptimizer(_SlaClipBase):

    def _update_threshold(self, C_t: float, s_hat: torch.Tensor) -> float:
        s1 = float(s_hat[0].item())
        # Eq. (12): C_{t+1} = C_t * exp( eta * ( gamma - s1 ) )
        C_next = float(C_t * math.exp(self.eta * (self.gamma - s1)))
        return float(max(self.c_min, min(self.c_max, C_next)))
