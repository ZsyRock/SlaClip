from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch

from .optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed


class AutoClipDPOptimizer(DPOptimizer):
    """
    Implements "Automatic clipping" from:
    Bu, Z., Wang, Y. X., Zha, S., & Karypis, G. (NeurIPS 2023). Automatic clipping: Differentially private deep learning made easier and stronger. 
    Paper: https://arxiv.org/abs/2206.07136
    """

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: Optional[int],
        # --- compatibility args (ignored by paper algorithm) ---
        autoclip_q: float = 0.5,
        ema_beta: float = 0.9,
        error_probe_enabled: bool = False,
        # --- paper-like args ---
        gamma: float = 0.01,
        mode: str = "auto_s",  # "auto_s" or "auto_v"
        secure_mode: bool = False,
        generator=None,
        loss_reduction: str = "mean",
        **kwargs,
    ):
        # AutoClip is a GLOBAL mechanism. If a per-layer list is accidentally passed in,
        # we sanitize it to a scalar to avoid DPOptimizer/add_noise interpreting it as
        # per-layer clipping semantics.
        if isinstance(max_grad_norm, (list, tuple)):
            R = float(max_grad_norm[0]) if len(max_grad_norm) > 0 else 1.0
        else:
            R = float(max_grad_norm)

        super().__init__(
            optimizer=optimizer,
            noise_multiplier=float(noise_multiplier),
            max_grad_norm=float(R),
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        # Keep old args to avoid surprising "unused argument" errors in callers
        self.autoclip_q = float(autoclip_q)
        self.ema_beta = float(ema_beta)
        self.error_probe_enabled = bool(error_probe_enabled)

        mode = str(mode).lower().strip()
        if mode not in ("auto_s", "auto_v"):
            raise ValueError(f"Unexpected AutoClip mode: {mode}. Use 'auto_s' or 'auto_v'.")
        self.mode = mode

        g = float(gamma)
        if self.mode == "auto_v":
            # AUTO-V corresponds to gamma=0, but we still use a tiny eps for numerical stability.
            g = 0.0
        if g < 0:
            raise ValueError("gamma must be >= 0.")
        self.gamma = g
        self._eps = 1e-12  # numerical stability for AUTO-V / near-zero norms

        # Optional diagnostics (research-only)
        self._last_per_sample_norms: Optional[torch.Tensor] = None
        self.error_probe_epoch_stats = {}

    def _compute_per_sample_norms(self) -> Optional[torch.Tensor]:
        """
        Compute per-sample L2 norms across all parameters.
        Returns a tensor of shape [B] on the same device as grad_samples[0],
        or None if no grad_samples are present.
        """
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return None

        # Empty batch (Poisson sampling can produce it)
        if len(self.grad_samples[0]) == 0:
            return self.grad_samples[0].new_zeros((0,))

        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]
        if not per_param_norms:
            return None
        return torch.stack(per_param_norms, dim=1).norm(2, dim=1)

    def clip_and_accumulate(self):
        """
        Override DPOptimizer.clip_and_accumulate with AutoClip scaling:

          scale_i = R / (||g_i|| + gamma)   [AUTO-S]
          scale_i = R / (||g_i|| + eps)     [AUTO-V]

        No clamp() (this is NOT Abadi clipping). The resulting ||ĝ_i||_2 < R holds.
        """
        norms = self._compute_per_sample_norms()
        if norms is None:
            # No parameters / no grad samples: behave like base
            return super().clip_and_accumulate()

        if norms.numel() == 0:
            per_sample_scale = norms  # shape [0]
        else:
            denom = norms + (self.gamma if self.gamma > 0 else self._eps)
            R = float(self.max_grad_norm)  # sanitized scalar
            per_sample_scale = (R / denom).to(dtype=norms.dtype)

        # Optional research-only diagnostics (do not enable in strict runs)
        if self.error_probe_enabled:
            self._last_per_sample_norms = norms.detach()
            if norms.numel() > 0:
                self.error_probe_epoch_stats = {
                    "mean_norm": float(norms.mean().item()),
                    "median_norm": float(norms.median().item()),
                    "max_norm": float(norms.max().item()),
                    "gamma": float(self.gamma),
                    "R": float(self.max_grad_norm),
                }
            else:
                self.error_probe_epoch_stats = {
                    "mean_norm": float("nan"),
                    "median_norm": float("nan"),
                    "max_norm": float("nan"),
                    "gamma": float(self.gamma),
                    "R": float(self.max_grad_norm),
                }

        # Accumulate: sum_i scale_i * g_i for each parameter
        for p in self.params:
            _check_processed_flag(p.grad_sample)

            # Keep computation in grad_sample dtype (typically fp32), do NOT cast to p.dtype
            grad_sample = self._get_flat_grad_sample(p)
            scale = per_sample_scale.to(device=grad_sample.device, dtype=grad_sample.dtype)
            grad = torch.einsum("i,i...", scale, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    # step() / pre_step() / add_noise() / scale_grad() are inherited from DPOptimizer.
    # Noise std remains: noise_multiplier * R, which matches the sensitivity bound.
