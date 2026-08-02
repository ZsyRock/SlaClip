"""SlaClip optimizer."""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch.optim import Optimizer

from .optimizer import DPOptimizer, _check_processed_flag, _generate_noise, _mark_as_processed

logger = logging.getLogger(__name__)

_STATE_DICT_KEY = "_slaclip_controller_state"


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
        c_max: float = 20.0,
        strict_paper_check: bool = True,
    ):
        if strict_paper_check and loss_reduction != "mean":
            raise ValueError(
                "strict_paper_check requires loss_reduction='mean' so the joint "
                "release is the fixed-denominator average in Eq. (9)-(11)"
            )
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
        self.c_min = float(c_min)
        self.c_max = float(c_max)

        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and > 0")
        if not math.isfinite(self.c_min) or self.c_min <= 0:
            raise ValueError("c_min must be finite and > 0")
        if not math.isfinite(self.c_max) or self.c_max < self.c_min:
            raise ValueError("c_max must be finite and >= c_min")
        if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be finite and > 0")
        if not math.isfinite(self.beta) or not 0.01 <= self.beta <= 0.99:
            raise ValueError("beta must be finite and in [0.01, 0.99]")
        if not math.isfinite(self.gamma) or not 0.01 <= self.gamma <= 0.99:
            raise ValueError("gamma must be finite and in [0.01, 0.99]")
        if self.strict_paper_check and not math.isclose(
            self.beta, 0.5, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "strict_paper_check requires beta=0.5, as fixed in Appendix C"
            )
        if self.strict_paper_check and not math.isclose(
            self.gamma, 0.5, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "strict_paper_check requires gamma=0.5 for the published SlaClip-Q median"
            )
        if self.strict_paper_check and not (
            math.isclose(self.c_min, 0.1, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(self.c_max, 20.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "strict_paper_check requires the declared reproduction guardrails "
                "c_min=0.1 and c_max=20.0"
            )
        if self.strict_paper_check and not (
            self.c_min <= float(max_grad_norm) <= self.c_max
        ):
            raise ValueError(
                "strict_paper_check requires max_grad_norm/C0 inside [0.1, 20.0]"
            )

        # C bounds are an explicit numerical guardrail used by the reproduction.
        # Clamp C_0 as well as later thresholds so the invariant holds from step 0.
        self.current_clip = float(
            max(self.c_min, min(self.c_max, float(max_grad_norm)))
        )
        self.max_grad_norm = self.current_clip

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

            # ``flatten(start_dim=1)`` keeps the feature width well-defined for
            # an empty Poisson batch, whereas ``view(0, -1)`` is ambiguous.
            g2 = flat.flatten(start_dim=1)
            sum_sq = sum_sq + (g2 * g2).sum(dim=1)

        if B is None:
            return
        batch_size = int(B)

        C_t = float(self.current_clip)
        eps = 1e-12
        # sqrt(0) is well-defined. Adding epsilon inside the square root would
        # turn an exact zero gradient into norm 1e-6 and perturb the near-zero
        # slack coordinate that drives Appendix C's controller.
        norms = torch.sqrt(sum_sq)

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

    def _update_threshold(self, C_t: float, s_hat: torch.Tensor) -> float:
        raise NotImplementedError

    def _bounded_threshold_update(self, C_t: float, exponent: float) -> float:
        """Compute ``clip(C_t * exp(exponent), c_min, c_max)`` safely."""
        upper_exponent = math.log(self.c_max / C_t)
        lower_exponent = math.log(self.c_min / C_t)
        if exponent >= upper_exponent:
            return self.c_max
        if exponent <= lower_exponent:
            return self.c_min
        return float(C_t * math.exp(exponent))

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

    def state_dict(self):
        """Save wrapped-optimizer and SlaClip controller state.

        Opacus does not serialize an in-flight virtual batch's ``summed_grad``.
        Refuse such checkpoints instead of producing a state that cannot resume
        the same logical DP-SGD step.
        """
        if self._is_last_step_skipped or self._has_unprocessed_private_state():
            raise RuntimeError(
                "Cannot checkpoint SlaClip with an unfinished private query; finish "
                "backward and all physical microbatches through optimizer.step() first."
            )

        state = super().state_dict()
        state[_STATE_DICT_KEY] = {
            "version": 1,
            "optimizer_class": type(self).__name__,
            "current_clip": float(self.current_clip),
            "K": int(self.K),
            "eta": float(self.eta),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "c_min": float(self.c_min),
            "c_max": float(self.c_max),
            "strict_paper_check": bool(self.strict_paper_check),
        }
        return state

    def _has_unprocessed_private_state(self) -> bool:
        """Detect a backward/clip result that has not completed its DP release."""
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
        """Restore a logical-step-boundary SlaClip checkpoint."""
        controller_state = state_dict.get(_STATE_DICT_KEY)
        optimizer_state = {
            key: value for key, value in state_dict.items() if key != _STATE_DICT_KEY
        }
        super().load_state_dict(optimizer_state)

        # Backwards compatibility for checkpoints made before controller state
        # was added: keep the constructor-provided clipping configuration.
        if controller_state is None:
            return
        if int(controller_state.get("version", -1)) != 1:
            raise ValueError("Unsupported SlaClip controller checkpoint version")
        if controller_state.get("optimizer_class") != type(self).__name__:
            raise ValueError("SlaClip checkpoint optimizer class does not match")

        expected_configuration = {
            "K": int(self.K),
            "eta": float(self.eta),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "c_min": float(self.c_min),
            "c_max": float(self.c_max),
            "strict_paper_check": bool(self.strict_paper_check),
        }
        for key, expected in expected_configuration.items():
            if controller_state.get(key) != expected:
                raise ValueError(
                    f"SlaClip checkpoint configuration mismatch for {key}: "
                    f"checkpoint={controller_state.get(key)!r}, current={expected!r}"
                )

        restored_clip = float(controller_state["current_clip"])
        if not (
            math.isfinite(restored_clip)
            and self.c_min <= restored_clip <= self.c_max
        ):
            raise ValueError(
                "SlaClip checkpoint current_clip lies outside configured bounds"
            )

        self.current_clip = restored_clip
        self.max_grad_norm = restored_clip
        self._sample_count = 0
        self._slack_sum = None
        self._lambda_t = 0.0
        self._slack_indicator = None


class SlaClipOptimizer(_SlaClipBase):

    def _update_threshold(self, C_t: float, s_hat: torch.Tensor) -> float:
        # Full SlaClip adaptive-threshold update.
        if self.strict_paper_check:
            if s_hat.numel() != int(self.K):
                raise ValueError("strict_paper_check: slack_indicator length != K")
        s_first = float(s_hat[0].item())
        s_last = float(s_hat[int(self.K) - 1].item())

        # Appendix C, Eqs. (28)-(30): first clip the normalized last slot,
        # then map it to gamma_t. In paper mode the coefficient is fixed at 1/2.
        r_t = float(max(0.0, min(1.0, s_last / C_t)))
        feedback_weight = 0.5 if self.strict_paper_check else self.beta
        gamma_t = 1.0 - feedback_weight * (1.0 - r_t)
        gamma_t = float(max(0.0, min(1.0, gamma_t)))

        exponent = self.eta * (gamma_t - s_first)
        return self._bounded_threshold_update(C_t, exponent)


class SlaClipQOptimizer(_SlaClipBase):

    def _update_threshold(self, C_t: float, s_hat: torch.Tensor) -> float:
        s1 = float(s_hat[0].item())
        # Eq. (12): C_{t+1} = C_t * exp( eta * ( gamma - s1 ) )
        exponent = self.eta * (self.gamma - s1)
        return self._bounded_threshold_update(C_t, exponent)
