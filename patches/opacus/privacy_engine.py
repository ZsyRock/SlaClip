#!/usr/bin/env python3
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

import inspect
import os
import warnings
from itertools import chain
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union

import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import (
    AbstractGradSampleModule,
    GradSampleModule,
    get_gsm_class,
    wrap_model,
)
from opacus.optimizers import DPOptimizer, get_optimizer_class
from opacus.optimizers import SlaClipOptimizer, SlaClipQOptimizer
from opacus.optimizers.autoclipoptimizer import AutoClipDPOptimizer
from opacus.schedulers import _GradClipScheduler, _NoiseScheduler
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.validators.module_validator import ModuleValidator
from torch import nn, optim
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def _filter_kwargs_for_init(init_fn, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Best-effort filtering: if target __init__ supports **kwargs, keep all.
    Otherwise only pass parameters that exist in the signature.
    Returns (filtered_kwargs, dropped_keys).
    """
    try:
        sig = inspect.signature(init_fn)
    except Exception:
        return dict(kwargs), []

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []

    allowed = set(params.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = [k for k in kwargs.keys() if k not in allowed]
    return filtered, dropped


class PrivacyEngine:
    """Entry point for Opacus DP training."""

    def __init__(
        self,
        *,
        accountant: str = "rdp",
        secure_mode: bool = False,
        privacy_mode: str = "strict",
    ):
        """Create a PrivacyEngine."""
        if privacy_mode not in ("strict", "research"):
            raise ValueError(f"Unexpected privacy_mode: {privacy_mode}")
        self.privacy_mode = privacy_mode

        self.accountant = create_accountant(mechanism=accountant)
        self.secure_mode = secure_mode
        self.secure_rng = None
        self.dataset = None  # only used to detect switching to a different dataset

        self.sample_rate: Optional[float] = None
        self.noise_multiplier: Optional[float] = None
        self.max_grad_norm: Optional[Union[float, List[float]]] = None

        if self.secure_mode:
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.secure_rng = csprng.create_random_device_generator("/dev/urandom")
        else:
            warnings.warn(
                "Secure RNG turned off. This is fine for experimentation (faster), "
                "but remember to turn it on and retrain one last time before production "
                "with ``secure_mode=True``.",
                stacklevel=2,
            )

    def _enforce_privacy_mode_guardrails(
        self,
        *,
        clipping: str,
        poisson_sampling: bool,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        In strict mode, block switches that may invalidate DP guarantees.
        """
        if self.privacy_mode != "strict":
            return

        if bool(kwargs.get("expose_true_hist", False)):
            raise ValueError("[privacy_mode=strict] expose_true_hist is research-only.")

        if bool(kwargs.get("error_probe_enabled", False)):
            raise ValueError("[privacy_mode=strict] error_probe_enabled is research-only.")

        if not poisson_sampling:
            warnings.warn(
                "[privacy_mode=strict] poisson_sampling=False does not match the Poisson "
                "sampling assumption used by standard DP accountants. Results may be an "
                "approximation unless your pipeline justifies this setting.",
                stacklevel=2,
            )

    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        if clipping in ("slaclip", "slaclip-q"):
            target_cls = SlaClipOptimizer if clipping == "slaclip" else SlaClipQOptimizer
            sl_kwargs, dropped = _filter_kwargs_for_init(target_cls.__init__, kwargs)
            for k in (
                "optimizer",
                "noise_multiplier",
                "max_grad_norm",
                "expected_batch_size",
                "loss_reduction",
                "generator",
                "secure_mode",
            ):
                sl_kwargs.pop(k, None)
            if dropped:
                warnings.warn(
                    f"[PrivacyEngine] SlaClip optimizer ignored unknown kwargs: {dropped}",
                    stacklevel=2,
                )

            return target_cls(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
                generator=generator,
                secure_mode=self.secure_mode,
                **sl_kwargs,
            )

        if clipping == "autoclip":
            ac_kwargs, dropped = _filter_kwargs_for_init(AutoClipDPOptimizer.__init__, kwargs)
            for k in (
                "optimizer",
                "noise_multiplier",
                "max_grad_norm",
                "expected_batch_size",
                "loss_reduction",
                "generator",
                "secure_mode",
                "autoclip_q",
                "error_probe_enabled",
            ):
                ac_kwargs.pop(k, None)

            if dropped:
                warnings.warn(
                    f"[PrivacyEngine] AutoClipDPOptimizer ignored unknown kwargs: {dropped}",
                    stacklevel=2,
                )

            return AutoClipDPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                autoclip_q=kwargs.get("autoclip_q", 0.5),
                error_probe_enabled=kwargs.get("error_probe_enabled", False),
                secure_mode=self.secure_mode,
                generator=generator,
                loss_reduction=loss_reduction,
                **ac_kwargs,
            )

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            **kwargs,
        )

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        *,
        poisson_sampling: bool,
        distributed: bool,
        generator=None,
    ) -> DataLoader:
        if self.dataset is None:
            self.dataset = data_loader.dataset
        elif self.dataset != data_loader.dataset:
            warnings.warn(
                f"PrivacyEngine detected new dataset object. "
                f"Was: {self.dataset}, got: {data_loader.dataset}. "
                f"Privacy accounting works per dataset, please initialize "
                f"new PrivacyEngine if you're using different dataset. "
                f"You can ignore this warning if two datasets above "
                f"represent the same logical dataset",
                stacklevel=2,
            )

        dl_gen = self.secure_rng if self.secure_mode else generator

        if poisson_sampling:
            return DPDataLoader.from_data_loader(
                data_loader, generator=dl_gen, distributed=distributed
            )

        if dl_gen is not None:
            try:
                return switch_generator(data_loader=data_loader, generator=dl_gen)
            except Exception as e:
                warnings.warn(
                    f"[PrivacyEngine] switch_generator failed, fallback to original data_loader. Error: {e}",
                    stacklevel=2,
                )
                return data_loader

        return data_loader

    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        max_grad_norm: Union[float, List[float]] = 1.0,
        loss_reduction: str = "mean",
        grad_sample_mode: str = "hooks",
    ) -> AbstractGradSampleModule:
        self.validate(module=module, optimizer=None, data_loader=None)

        if isinstance(module, AbstractGradSampleModule):
            if (
                module.batch_first != batch_first
                or module.loss_reduction != loss_reduction
                or type(module) is not get_gsm_class(grad_sample_mode)
            ):
                raise ValueError(
                    "Pre-existing GradSampleModule doesn't match new arguments. "
                    f"Got: module.batch_first={module.batch_first}, "
                    f"module.loss_reduction={module.loss_reduction}, "
                    f"type(module)={type(module)}. "
                    f"Requested: batch_first={batch_first}, loss_reduction={loss_reduction}, "
                    f"grad_sample_mode={grad_sample_mode}. "
                    "Please pass vanilla nn.Module instead."
                )
            return module

        if grad_sample_mode in ["ghost", "ghost_fsdp"]:
            return wrap_model(
                module,
                grad_sample_mode=grad_sample_mode,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
                max_grad_norm=max_grad_norm,
            )

        return wrap_model(
            module,
            grad_sample_mode=grad_sample_mode,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

    def _prepare_criterion(
        self,
        *,
        module: GradSampleModule,
        optimizer: DPOptimizer,
        criterion=nn.CrossEntropyLoss(),
        loss_reduction: str = "mean",
        **kwargs,
    ) -> DPLossFastGradientClipping:
        return DPLossFastGradientClipping(module, optimizer, criterion, loss_reduction)

    def is_compatible(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ) -> bool:
        return ModuleValidator.is_valid(module)

    def validate(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ):
        ModuleValidator.validate(module, strict=True)

    @classmethod
    def get_compatible_module(cls, module: nn.Module) -> nn.Module:
        module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, strict=True)
        return module

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # default for backward compatibility
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ) -> Union[
        Tuple[GradSampleModule, DPOptimizer, DataLoader],
        Tuple[GradSampleModule, DPOptimizer, DPLossFastGradientClipping, DataLoader],
    ]:
        if noise_generator and self.secure_mode:
            raise ValueError("Passing noise_generator is prohibited in secure mode")

        # Guardrails first (before constructing optimizer wrappers)
        self._enforce_privacy_mode_guardrails(
            clipping=clipping, poisson_sampling=poisson_sampling, kwargs=kwargs
        )

        # Compare module parameters with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError("Module parameters are different than optimizer Parameters")

        distributed = isinstance(module, (DPDDP, DDP, FSDPModule))

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.forbid_grad_accumulation()

        data_loader = self._prepare_data_loader(
            data_loader,
            distributed=distributed,
            poisson_sampling=poisson_sampling,
            generator=noise_generator,
        )

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            expected_batch_size = int(expected_batch_size / max(1, int(world_size)))

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            **kwargs,
        )

        self.sample_rate = float(sample_rate)
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = max_grad_norm

        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )

        if "ghost" in grad_sample_mode:
            criterion = self._prepare_criterion(
                module=module,
                optimizer=optimizer,
                criterion=criterion,
                loss_reduction=loss_reduction,
                **kwargs,
            )
            return module, optimizer, criterion, data_loader

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # default for backward compatibility
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ) -> Union[
        Tuple[GradSampleModule, DPOptimizer, DataLoader],
        Tuple[GradSampleModule, DPOptimizer, DPLossFastGradientClipping, DataLoader],
    ]:
        sample_rate = 1 / len(data_loader)

        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher.",
                stacklevel=2,
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            criterion=criterion,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant=self.accountant.mechanism(),
                **kwargs,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            **kwargs,
        )

    def get_epsilon(self, delta):
        return self.accountant.get_epsilon(delta)

    def save_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModule,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        module_state_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_save_kwargs: Optional[Dict[str, Any]] = None,
    ):
        checkpoint_dict = checkpoint_dict or {}
        checkpoint_dict["module_state_dict"] = module.state_dict(
            **(module_state_dict_kwargs or {})
        )
        checkpoint_dict["privacy_accountant_state_dict"] = self.accountant.state_dict()
        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
        if noise_scheduler is not None:
            checkpoint_dict["noise_scheduler_state_dict"] = noise_scheduler.state_dict()
        if grad_clip_scheduler is not None:
            checkpoint_dict["grad_clip_scheduler_state_dict"] = (
                grad_clip_scheduler.state_dict()
            )

        torch.save(checkpoint_dict, path, **(torch_save_kwargs or {}))

    def load_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModule,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        module_load_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        checkpoint = torch.load(path, **(torch_load_kwargs or {}), weights_only=False)
        module.load_state_dict(
            checkpoint["module_state_dict"], **(module_load_dict_kwargs or {})
        )
        self.accountant.load_state_dict(checkpoint["privacy_accountant_state_dict"])

        optimizer_state_dict = checkpoint.pop("optimizer_state_dict", {})
        if optimizer is not None and len(optimizer_state_dict) > 0:
            optimizer.load_state_dict(optimizer_state_dict)
        elif (optimizer is not None) ^ (len(optimizer_state_dict) > 0):
            warnings.warn(
                f"optimizer_state_dict has {len(optimizer_state_dict)} items"
                f" but optimizer is {'' if optimizer else 'not'} provided.",
                stacklevel=2,
            )

        noise_scheduler_state_dict = checkpoint.pop("noise_scheduler_state_dict", {})
        if noise_scheduler is not None and len(noise_scheduler_state_dict) > 0:
            noise_scheduler.load_state_dict(noise_scheduler_state_dict)

        grad_clip_scheduler_state_dict = checkpoint.pop(
            "grad_clip_scheduler_state_dict", {}
        )
        if grad_clip_scheduler is not None and len(grad_clip_scheduler_state_dict) > 0:
            grad_clip_scheduler.load_state_dict(grad_clip_scheduler_state_dict)

        return checkpoint
