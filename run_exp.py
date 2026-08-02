#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_PATCHES_DIR = _THIS_DIR / "patches"

# The repository is an exact Opacus checkout with SlaClip kept as a nested,
# versioned overlay. The overlay must win for patched Opacus modules.
sys.path.insert(0, str(_PATCHES_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(1, str(_THIS_DIR))

import numpy as np
import torch
import torch.optim as optim

try:
    import opacus
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except Exception as exc:
    raise RuntimeError(
        f"Failed to import the local Opacus/SlaClip overlay: {exc}. Run from Opacus-Aug, "
        "install it editable, then execute `python SlaClip/verify_install.py`."
    ) from exc

from slaclip.args import (
    PAPER_PRACTICAL_K_SIGMA1,
    paper_k_upper_bound,
    paper_recommended_k,
    parse_args,
)
from slaclip.data import make_dataloaders
from slaclip.logging_utils import (
    assert_output_paths_available,
    collect_run_metadata,
    ensure_output_dir,
    write_config_json,
    write_epoch_csv,
    write_run_json,
)
from slaclip.models import make_model
from slaclip.protocol import apply_protocol
from slaclip.train_loop import build_epoch_record, evaluate, train_one_epoch


def _method_uses_k(method: str) -> bool:
    return str(method).lower().strip() in {"slaclip", "slaclip-q"}


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _assert_local_opacus() -> None:
    opacus_path = Path(opacus.__file__).resolve()
    privacy_engine_path = Path(opacus.privacy_engine.__file__).resolve()
    print(f"[SlaClip] opacus.__file__ = {opacus_path}")
    print(f"[SlaClip] opacus.privacy_engine = {privacy_engine_path}")

    if "site-packages" in str(opacus_path) or "dist-packages" in str(opacus_path):
        raise RuntimeError(
            "Opacus was imported from site-packages, not Opacus-Aug. Install Opacus-Aug "
            "editable and run from its repository root."
        )
    try:
        opacus_path.relative_to(_REPO_ROOT)
        privacy_engine_path.relative_to(_PATCHES_DIR)
    except ValueError as exc:
        raise RuntimeError(
            "The active Opacus/SlaClip code is not the versioned local overlay: "
            f"opacus={opacus_path}, privacy_engine={privacy_engine_path}."
        ) from exc


def _resolve_device(requested: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    print(f"[SlaClip] torch.cuda.is_available() = {cuda_available}")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but is unavailable to this PyTorch build"
        )
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_available else "cpu")


def _build_optimizer(args, model):
    if args.optim == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    if args.optim == "Adam":
        return optim.Adam(
            model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
    if args.optim == "RMSprop":
        return optim.RMSprop(
            model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
    raise ValueError(f"Unsupported optimizer: {args.optim}")


def _build_lr_scheduler(args, optimizer):
    if args.lr_schedule == "constant":
        return None
    if args.lr_schedule == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.epochs)
        )
    raise ValueError(f"Unsupported learning-rate schedule: {args.lr_schedule}")


def _configure_noise_and_k(args, train_loader) -> dict:
    logical_steps = int(len(train_loader))
    if logical_steps <= 0:
        raise ValueError("Training loader contains no logical steps")
    sample_rate = 1.0 / float(logical_steps)
    train_size = int(len(train_loader.dataset))
    expected_batch_size = int(train_size * sample_rate)
    if expected_batch_size <= 0:
        raise ValueError("Expected logical batch size rounded to zero")

    if args.method != "nondp" and args.epsilon_mode == "calibrate":
        args.sigma = float(
            get_noise_multiplier(
                target_epsilon=float(args.target_epsilon),
                target_delta=float(args.delta),
                sample_rate=sample_rate,
                epochs=int(args.epochs),
                accountant=str(args.accountant),
                epsilon_tolerance=float(args.epsilon_tolerance),
            )
        )
        print(
            f"[SlaClip] calibrated sigma={args.sigma:.12f} for "
            f"epsilon={args.target_epsilon:g}, delta={args.delta:g}, "
            f"q_eff={sample_rate:.12f}, steps={logical_steps * int(args.epochs)}"
        )
    elif args.method != "nondp" and args.sigma is None:
        raise ValueError("A private run requires a fixed or calibrated sigma")

    k_metadata = None
    if _method_uses_k(args.method):
        upper_bound = paper_k_upper_bound(expected_batch_size, float(args.sigma))
        source = "explicit"
        if args.K is None:
            practical_k = PAPER_PRACTICAL_K_SIGMA1.get(int(args.batch_size))
            if (
                args.protocol == "controlled"
                and math.isclose(float(args.sigma), 1.0, rel_tol=0.0, abs_tol=1e-12)
                and practical_k is not None
            ):
                # Appendix D/Table 3 gives convenient experiment defaults below
                # the Eq. (14) bound for the controlled sigma=1 recipe.
                args.K = int(practical_k)
                source = "Appendix D Table 3 practical sigma=1"
            else:
                args.K = paper_recommended_k(expected_batch_size, float(args.sigma))
                source = "Eq.14 floor"
        k_metadata = {
            "source": source,
            "K": int(args.K),
            "K_max_99": float(upper_bound),
            "within_eq14_bound": bool(int(args.K) <= math.floor(upper_bound)),
            "expected_batch_size_used": int(expected_batch_size),
            "sigma_used": float(args.sigma),
        }
        print(
            f"[SlaClip] K={int(args.K)} ({source}); Eq.14 K_max={upper_bound:.6f}, "
            f"expected_B={expected_batch_size}, sigma={float(args.sigma):.8g}"
        )

    return {
        "logical_steps_per_epoch": logical_steps,
        "sample_rate": sample_rate,
        "expected_batch_size": expected_batch_size,
        "K_selection": k_metadata,
    }


def _make_private(args, model, optimizer, criterion, train_loader):
    method = str(args.method).lower().strip()
    if method == "nondp":
        return None, model, optimizer, criterion, train_loader

    try:
        privacy_engine = PrivacyEngine(
            accountant=str(args.accountant),
            secure_mode=bool(args.secure_mode),
        )
    except Exception as exc:
        if args.secure_mode:
            raise RuntimeError(
                "Opacus secure_mode could not initialize. Install/configure torchcsprng; "
                "do not silently fall back for a run requested as secure."
            ) from exc
        raise

    clipping_by_method = {
        "slaclip": "slaclip",
        "slaclip-q": "slaclip-q",
        "vanilla-clip": "flat",
        "adap-clip": "adaptive",
        "dc-sgd-e": "dc-sgd-e",
        "autoclip": "autoclip",
    }
    clipping = clipping_by_method.get(method)
    if clipping is None:
        raise ValueError(f"Unsupported method: {method}")

    kwargs = {
        "module": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "data_loader": train_loader,
        "noise_multiplier": float(args.sigma),
        "max_grad_norm": float(args.C0),
        "clipping": clipping,
        "grad_sample_mode": str(args.grad_sample_mode),
        "poisson_sampling": True,
    }
    if method in {"slaclip", "slaclip-q"}:
        kwargs.update(
            {
                "num_slots": int(args.K),
                "eta": float(args.eta),
                "beta": float(args.beta),
                "gamma": float(args.gamma),
                "c_min": float(args.c_min),
                "c_max": float(args.c_max),
                "strict_paper_check": bool(args.strict_paper_check),
            }
        )
    elif method == "adap-clip":
        # The released paper implementation uses the nominal logical B/20.
        sigma_b = max(1.0, float(args.batch_size) / 20.0)
        args.adap_count_noise_std = float(sigma_b)
        kwargs.update(
            {
                "target_unclipped_quantile": float(args.gamma),
                "clipbound_learning_rate": float(args.eta),
                "max_clipbound": float(args.c_max),
                "min_clipbound": float(args.c_min),
                "unclipped_num_std": sigma_b,
            }
        )
    elif method == "dc-sgd-e":
        kwargs.update(
            {
                "batchsize_train": int(args.batch_size),
                "dimension": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "percentile": float(args.dc_percentile),
                "stride": float(args.dc_stride),
                "bin_cnt": int(args.dc_bin_count),
                "histogram_std": float(args.dc_histogram_std),
                "c_min": float(args.c_min),
                "c_max": float(args.c_max),
            }
        )

    result = privacy_engine.make_private(**kwargs)
    if "ghost" in str(args.grad_sample_mode):
        model, optimizer, criterion, train_loader = result
    else:
        model, optimizer, train_loader = result
    return privacy_engine, model, optimizer, criterion, train_loader


def _get_clip_value(optimizer) -> float:
    for attribute in ("current_clip", "max_grad_norm"):
        value = getattr(optimizer, attribute, None)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _get_epsilon(privacy_engine, delta: float) -> float:
    if privacy_engine is None:
        return float("nan")
    return float(privacy_engine.accountant.get_epsilon(delta=float(delta)))


def _resolved_method_configuration(args, optimizer) -> dict:
    """Record the mechanism actually instantiated, including hidden defaults."""

    method = str(args.method)
    if method == "nondp":
        return {
            "method": method,
            "private": False,
            "optimizer_class": type(optimizer).__name__,
        }

    accountant_noise_multiplier = getattr(
        optimizer,
        "_accountant_noise_multiplier",
        getattr(optimizer, "noise_multiplier", None),
    )
    try:
        accountant_noise_multiplier = float(accountant_noise_multiplier)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "The instantiated private optimizer does not expose its accountant "
            "noise multiplier"
        ) from exc
    initial_clip = _get_clip_value(optimizer)
    if not math.isfinite(accountant_noise_multiplier) or not math.isfinite(
        initial_clip
    ):
        raise RuntimeError(
            "The instantiated private optimizer exposes non-finite mechanism parameters"
        )

    resolved = {
        "method": method,
        "private": True,
        "optimizer_class": type(optimizer).__name__,
        "loss_reduction": str(getattr(optimizer, "loss_reduction", "mean")),
        "accountant_noise_multiplier": accountant_noise_multiplier,
        "initial_C0": initial_clip,
        "c_min": float(args.c_min),
        "c_max": float(args.c_max),
    }
    if method in {"slaclip", "slaclip-q"}:
        resolved.update(
            {
                "K": int(optimizer.K),
                "eta": float(optimizer.eta),
                "beta": float(optimizer.beta),
                "gamma": float(optimizer.gamma),
                "lambda_initial": float(args.C0) / math.sqrt(int(optimizer.K)),
                "joint_release": "one isotropic d+K Gaussian vector",
                "controller": (
                    "Appendix C Eq.28-30" if method == "slaclip" else "Eq.12 median"
                ),
            }
        )
    elif method == "adap-clip":
        resolved.update(
            {
                "target_unclipped_quantile": float(optimizer.target_unclipped_quantile),
                "clipbound_learning_rate": float(optimizer.clipbound_learning_rate),
                "count_noise_std": float(optimizer.unclipped_num_std),
                "gradient_noise_multiplier": float(optimizer._grad_noise_multiplier),
                "privacy_composition": "Andrew et al. Theorem 1 split",
                "count_query_kind": "centered_bit",
                "count_query_definition": "1{||g_i||<=C}-1/2",
                "count_query_l2_sensitivity": 0.5,
                "count_release_denominator": (
                    "expected_batch_size*accumulated_iterations"
                ),
                "count_release_expected_batch_size": int(optimizer.expected_batch_size),
                "empty_poisson_count_release": True,
            }
        )
    elif method == "dc-sgd-e":
        resolved.update(
            {
                "histogram_noise_std": float(optimizer.histogram_std),
                "gradient_noise_multiplier": float(optimizer._grad_noise_multiplier),
                "percentile": float(optimizer.percentile),
                "stride_initial": float(args.dc_stride),
                "bin_count": int(optimizer.bin_cnt),
                "dimension": int(optimizer.dimension),
                "nominal_batch_size": int(optimizer.batchsize_train),
                "boundary_search_iteration_cap": int(
                    optimizer.max_boundary_search_iterations
                ),
                "empty_poisson_zero_histogram_release": True,
                "empty_poisson_histogram_input": "all-zero vector",
            }
        )
    elif method == "autoclip":
        resolved.update(
            {
                "mode": str(optimizer.mode),
                "auto_s_gamma": float(optimizer.gamma),
                "sensitivity_radius_R": float(optimizer.max_grad_norm),
                "transform": "R*g/(||g||+gamma), without Abadi clamp",
            }
        )
    elif method == "vanilla-clip":
        resolved["clipping"] = "global flat clipping"
    return resolved


def main() -> None:
    args = parse_args()
    protocol_metadata = apply_protocol(args)
    _set_seed(int(args.seed))
    _assert_local_opacus()
    device = _resolve_device(args.device)
    print(f"[SlaClip] using device={device}")

    out_dir_arg = Path(args.out_dir)
    out_dir = ensure_output_dir(
        str(out_dir_arg if out_dir_arg.is_absolute() else _THIS_DIR / out_dir_arg)
    )
    run_name = str(args.run_name).strip() or "slaclip_run"
    csv_path = out_dir / f"{run_name}.csv"
    json_path = out_dir / f"{run_name}.json"
    config_path = out_dir / f"{run_name}_config.json"
    assert_output_paths_available(
        [csv_path, json_path, config_path], overwrite=bool(args.overwrite)
    )

    (
        train_loader,
        validation_loader,
        test_loader,
        num_classes,
        data_metadata,
    ) = make_dataloaders(args)
    sampling_metadata = _configure_noise_and_k(args, train_loader)
    protocol_metadata["K_selection"] = sampling_metadata["K_selection"]

    model = make_model(args.dataset, num_classes, data_metadata, args).to(device)
    optimizer = _build_optimizer(args, model)
    scheduler = _build_lr_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    privacy_engine, model, optimizer, criterion, train_loader = _make_private(
        args, model, optimizer, criterion, train_loader
    )
    protocol_metadata["resolved_method"] = _resolved_method_configuration(
        args, optimizer
    )

    metadata = collect_run_metadata(
        args=args,
        protocol_metadata=protocol_metadata,
        data_metadata=data_metadata,
        opacus_root=_REPO_ROOT,
        slaclip_root=_THIS_DIR,
        device=device,
        logical_steps_per_epoch=sampling_metadata["logical_steps_per_epoch"],
        expected_batch_size=sampling_metadata["expected_batch_size"],
    )
    write_config_json(config_path, metadata)

    meta_record = {
        "dataset": str(args.dataset),
        "method": str(args.method),
        "protocol": str(args.protocol),
        "phase": str(args.phase),
        "seed": int(args.seed),
    }
    records: list[dict] = []
    cumulative_logical_steps = 0
    hard_stop = args.epsilon_mode == "hard-stop"
    clip_boundary_hits = {"min": 0, "max": 0}

    if privacy_engine is None:
        memory_context = nullcontext(train_loader)
    else:
        memory_context = BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=int(args.max_physical_batch_size),
            optimizer=optimizer,
        )

    with memory_context as memory_safe_loader:
        stopped_early = False
        for epoch in range(1, int(args.epochs) + 1):
            boundary_hits_before_epoch = dict(clip_boundary_hits)

            def on_logical_step(info):
                clip_value = _get_clip_value(info.get("optimizer"))
                if math.isfinite(clip_value):
                    if math.isclose(
                        clip_value,
                        float(args.c_min),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        clip_boundary_hits["min"] += 1
                    if math.isclose(
                        clip_value,
                        float(args.c_max),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        clip_boundary_hits["max"] += 1
                if not hard_stop:
                    return False
                epsilon = float(info.get("epsilon", float("nan")))
                return math.isfinite(epsilon) and epsilon >= float(args.target_epsilon)

            train_loss, train_accuracy, stopped_early, epoch_steps = train_one_epoch(
                model=model,
                optimizer=optimizer,
                loader=memory_safe_loader,
                device=device,
                criterion=criterion,
                epoch=epoch,
                privacy_engine=privacy_engine,
                delta=float(args.delta),
                on_batch_end=on_logical_step,
                expose_training_metrics=privacy_engine is None,
            )
            cumulative_logical_steps += int(epoch_steps)

            validation_loss = validation_accuracy = None
            test_loss = test_accuracy = None
            if args.phase == "selection":
                if validation_loader is None or test_loader is not None:
                    raise RuntimeError(
                        "Selection phase loader isolation invariant failed"
                    )
                validation_loss, validation_accuracy = evaluate(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    criterion=criterion,
                    split_tag="public_validation",
                )
            elif args.phase == "controlled":
                if test_loader is None:
                    raise RuntimeError("Controlled diagnostics require a test loader")
                test_loss, test_accuracy = evaluate(
                    model=model,
                    loader=test_loader,
                    device=device,
                    criterion=criterion,
                    split_tag="test_diagnostic",
                )

            record = build_epoch_record(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                logical_steps_completed=cumulative_logical_steps,
                privacy_engine=privacy_engine,
                delta=float(args.delta),
                meta=meta_record,
                C_t=_get_clip_value(optimizer),
            )
            record["C_min_hits"] = int(
                clip_boundary_hits["min"] - boundary_hits_before_epoch["min"]
            )
            record["C_max_hits"] = int(
                clip_boundary_hits["max"] - boundary_hits_before_epoch["max"]
            )
            record["C_boundary_hit"] = bool(
                record["C_min_hits"] or record["C_max_hits"]
            )
            records.append(record)
            write_epoch_csv(csv_path, records)
            write_run_json(json_path, metadata=metadata, records=records, final=None)

            if scheduler is not None:
                scheduler.step()
            if stopped_early:
                break

    # Main retraining keeps the official test split sealed until the prescribed
    # horizon is complete. Custom runs also default to one final test query.
    final_test_loss = final_test_accuracy = None
    if args.phase in {"retrain", "custom"} and test_loader is not None:
        final_test_loss, final_test_accuracy = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            split_tag="final_test",
        )
    elif args.phase == "controlled" and records:
        final_test_loss = records[-1]["test_loss"]
        final_test_accuracy = records[-1]["test_accuracy"]

    actual_epsilon = _get_epsilon(privacy_engine, float(args.delta))
    final = {
        "epochs_completed": int(len(records)),
        "logical_steps_completed": int(cumulative_logical_steps),
        "epsilon": actual_epsilon,
        "delta": float(args.delta),
        "target_epsilon": (
            float(args.target_epsilon) if args.target_epsilon is not None else None
        ),
        "epsilon_mode": str(args.epsilon_mode),
        "sigma": float(args.sigma) if args.sigma is not None else None,
        "final_validation_accuracy": (
            records[-1]["validation_accuracy"]
            if args.phase == "selection" and records
            else None
        ),
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
        "final_C_t": _get_clip_value(optimizer),
        "C_guardrails": {
            "min": float(args.c_min),
            "max": float(args.c_max),
            "min_hits": int(clip_boundary_hits["min"]),
            "max_hits": int(clip_boundary_hits["max"]),
            "boundary_hit": bool(
                clip_boundary_hits["min"] or clip_boundary_hits["max"]
            ),
        },
        "hard_stop_overshoot": (
            actual_epsilon - float(args.target_epsilon)
            if hard_stop and math.isfinite(actual_epsilon)
            else None
        ),
    }
    write_epoch_csv(csv_path, records)
    write_run_json(json_path, metadata=metadata, records=records, final=final)
    print(f"Done. Wrote {csv_path}, {json_path}, and {config_path}")


if __name__ == "__main__":
    main()
