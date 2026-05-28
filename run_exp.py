#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_PATCHES_DIR = _THIS_DIR / "patches"

sys.path.insert(0, str(_PATCHES_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))

import numpy as np 
import torch 
import torch.optim as optim 

try:  
    import opacus 
    from opacus import PrivacyEngine 
    from opacus.accountants.utils import get_noise_multiplier
except Exception as e:  
    raise RuntimeError(
        f"Failed to import Opacus/PrivacyEngine: {e}. Fix: run from the Opacus repo root, "
        "install deps, then `pip install -e .` and verify with `python SlaClip/verify_install.py`."
    ) from e

from slaclip.args import paper_recommended_k, parse_args 
from slaclip.data import make_dataloaders  
from slaclip.logging_utils import ensure_output_dir, write_epoch_csv, write_epoch_json  
from slaclip.models import make_model  
from slaclip.train_loop import build_epoch_record, evaluate, train_one_epoch  


_DEFAULT_BATCH = {"cifar10": 1024, "mnist": 512, "fmnist": 512, "names": 512, "imdb": 256}
_TARGET_EPSILONS = {
    "cifar10": [5, 7, 9],
    "mnist": [1, 2, 3],
    "fmnist": [1, 2, 3],
    "imdb": [2, 4, 6],
    "names": [2, 4, 5],
}
_MILESTONE_EPS = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.5, 2.0, 2.5, 3.0,
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
]


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
            "Opacus was imported from site-packages. Fix: run from the clean Opacus repo root, "
            "then `pip install -e .` and execute `python SlaClip/verify_install.py` to confirm paths."
        )

    try:
        if not opacus_path.is_relative_to(_REPO_ROOT):
            raise RuntimeError(
                f"Opacus import is not from the local repo root: {_REPO_ROOT}. "
                f"Resolved: {opacus_path}. Fix: run from repo root and use `pip install -e .` "
                "then re-run `python SlaClip/verify_install.py`."
            )
    except AttributeError:
        if not str(opacus_path).startswith(str(_REPO_ROOT)):
            raise RuntimeError(
                f"Opacus import is not from the local repo root: {_REPO_ROOT}. "
                f"Resolved: {opacus_path}. Fix: run from repo root and use `pip install -e .` "
                "then re-run `python SlaClip/verify_install.py`."
            )

    try:
        if not privacy_engine_path.is_relative_to(_PATCHES_DIR):
            raise RuntimeError(
                f"privacy_engine is not loaded from SlaClip patches: {_PATCHES_DIR}. "
                f"Resolved: {privacy_engine_path}. Fix: ensure SlaClip/patches is inserted "
                "before importing opacus, then re-run `python SlaClip/verify_install.py`."
            )
    except AttributeError:
        if not str(privacy_engine_path).startswith(str(_PATCHES_DIR)):
            raise RuntimeError(
                f"privacy_engine is not loaded from SlaClip patches: {_PATCHES_DIR}. "
                f"Resolved: {privacy_engine_path}. Fix: ensure SlaClip/patches is inserted "
                "before importing opacus, then re-run `python SlaClip/verify_install.py`."
            )


def _apply_defaults(args, *, set_k: bool = True) -> None:
    ds = str(args.dataset).lower().strip()

    if args.batch_size is None:
        if ds == "mnist":
            args.batch_size = 512
        else:
            args.batch_size = int(_DEFAULT_BATCH[ds])
    if args.batch_size_test is None:
        if ds in {"mnist", "fmnist"}:
            args.batch_size_test = 1024
        else:
            args.batch_size_test = int(args.batch_size)

    if args.slot_fb_beta is not None:
        args.beta = float(args.slot_fb_beta)
    if set_k and args.K is None and _method_uses_k(args.method):
        args.K = int(paper_recommended_k(int(args.batch_size), float(args.sigma)))

    if ds == "names":
        if args.lr is None:
            args.lr = 2.0
        if args.momentum is None:
            args.momentum = 0.0
        if args.weight_decay is None:
            args.weight_decay = 0.0
        if args.lr_schedule is None:
            args.lr_schedule = "constant"
        if args.delta == 1e-5:
            args.delta = 8e-5
    else:
        if args.lr is None:
            args.lr = 0.1
        if args.momentum is None:
            args.momentum = 0.9
        if args.weight_decay is None:
            args.weight_decay = 5e-4
        if args.lr_schedule is None:
            args.lr_schedule = "cos"

    if args.use_paper_budgets and args.target_epsilon is None:
        args.target_epsilon = float(_TARGET_EPSILONS[ds][0])


def _build_optimizer(args, model):
    if args.optim == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optim == "Adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optim}")


def _build_lr_scheduler(args, optimizer):
    if args.lr_schedule == "constant":
        return None
    if args.lr_schedule == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return None


def _make_private(args, model, optimizer, train_loader):
    method = str(args.method).lower().strip()
    if method == "nondp":
        return None, model, optimizer, train_loader

    privacy_engine = PrivacyEngine(
        accountant=str(args.accountant),
        secure_mode=False,
    )

    if method == "slaclip":
        clipping = "slaclip"
    elif method == "slaclip-q":
        clipping = "slaclip-q"
    elif method == "vanilla-clip":
        clipping = "flat"
    elif method == "adap-clip":
        clipping = "adaptive"
    elif method == "dc-sgd-e":
        clipping = "dc-sgd-e"
    elif method == "autoclip":
        clipping = "autoclip"
    else:
        raise ValueError(f"Unsupported method: {method}")

    make_private_kwargs = dict(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=float(args.sigma),
        max_grad_norm=float(args.C0),
        clipping=clipping,
        grad_sample_mode=str(args.grad_sample_mode),
        poisson_sampling=True,
    )

    if method in {"slaclip", "slaclip-q"}:
        make_private_kwargs.update(
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

    if method == "adap-clip":
        sigma_b = max(1.0, float(args.batch_size) / 20.0)
        make_private_kwargs.update(
            {
                "target_unclipped_quantile": float(args.gamma),
                "clipbound_learning_rate": float(args.eta),
                "max_clipbound": 50.0,
                "min_clipbound": 0.1,
                "unclipped_num_std": float(sigma_b),
            }
        )

    if method == "dc-sgd-e":
        total_dim = sum(p.numel() for p in model.parameters())
        make_private_kwargs.update(
            {
                "batchsize_train": int(args.batch_size),
                "dimension": int(total_dim),
                "percentile": 0.3,
                "stride": 1.0,
                "bin_cnt": 20,
                "histogram_std": 5.0,
            }
        )

    model, optimizer, train_loader = privacy_engine.make_private(**make_private_kwargs)
    return privacy_engine, model, optimizer, train_loader


def _maybe_calibrate_sigma(args, train_loader) -> None:
    if not bool(args.calibrate_sigma):
        return

    if args.target_epsilon is None:
        raise ValueError("--calibrate-sigma requires --target-epsilon")

    sample_rate = 1.0 / float(len(train_loader))
    sigma = get_noise_multiplier(
        target_epsilon=float(args.target_epsilon),
        target_delta=float(args.delta),
        sample_rate=sample_rate,
        epochs=int(args.epochs),
        accountant=str(args.accountant),
    )
    args.sigma = float(sigma)
    if args.K is None and _method_uses_k(args.method):
        args.K = int(paper_recommended_k(int(args.batch_size), float(args.sigma)))
    print(
        "[SlaClip] Calibrated sigma = "
        f"{float(args.sigma):.12f} for eps={float(args.target_epsilon)}, "
        f"delta={float(args.delta)}, epochs={int(args.epochs)}, "
        f"sample_rate={sample_rate:.12f}"
        + (f", K={int(args.K)}" if _method_uses_k(args.method) else "")
    )


def main() -> None:
    args = parse_args()
    _apply_defaults(args, set_k=False)
    _set_seed(args.seed)
    _assert_local_opacus()

    cuda_available = torch.cuda.is_available()
    print(f"[SlaClip] torch.cuda.is_available() = {cuda_available}")
    if args.device == "cuda":
        if not cuda_available:
            print("[SlaClip] CUDA requested but not available.")
            print("[SlaClip] Check driver/CUDA toolkit compatibility and PyTorch CUDA build.")
            raise SystemExit(1)
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")
    print(f"[SlaClip] Using device: {device}")

    train_loader, test_loader, num_classes, meta = make_dataloaders(args)
    _maybe_calibrate_sigma(args, train_loader)
    if args.K is None and _method_uses_k(args.method):
        args.K = int(paper_recommended_k(int(args.batch_size), float(args.sigma)))
    model = make_model(args.dataset, num_classes, meta, args).to(device)

    optimizer = _build_optimizer(args, model)
    scheduler = _build_lr_scheduler(args, optimizer)

    privacy_engine, model, optimizer, train_loader = _make_private(
        args, model, optimizer, train_loader
    )

    criterion = torch.nn.CrossEntropyLoss()

    out_dir = ensure_output_dir(os.path.join(_THIS_DIR, args.out_dir))
    run_name = str(args.run_name).strip() or "slaclip_run"
    csv_path = out_dir / f"{run_name}.csv"
    json_path = out_dir / f"{run_name}.json"

    meta_record = {
        "dataset": str(args.dataset),
        "method": str(args.method),
        "seed": int(args.seed),
    }

    def _get_clip_value(opt) -> float:
        if hasattr(opt, "current_clip"):
            try:
                return float(getattr(opt, "current_clip"))
            except Exception:
                pass
        mg = getattr(opt, "max_grad_norm", None)
        if isinstance(mg, (list, tuple)) and mg:
            try:
                return float(mg[0])
            except Exception:
                return float("nan")
        try:
            return float(mg)
        except Exception:
            return float("nan")

    def _get_epsilon() -> float:
        try:
            return float(privacy_engine.accountant.get_epsilon(delta=float(args.delta)))
        except Exception:
            return float("nan")

    records = []
    events = []
    milestones = list(_MILESTONE_EPS)
    milestone_idx = 0
    best_diff = float("inf")
    best_event = None
    last_under_099 = None
    for epoch in range(1, int(args.epochs) + 1):
        last_step_info = {"step": 0, "batch_acc": float("nan"), "running_acc": float("nan")}

        def on_batch_end(info):
            nonlocal last_step_info, milestone_idx, best_diff, best_event, last_under_099
            last_step_info = {
                "step": info.get("step", 0),
                "batch_acc": info.get("batch_acc", float("nan")),
                "running_acc": info.get("running_acc", float("nan")),
            }
            eps = float(info.get("epsilon", float("nan")))
            if not (eps == eps):
                return False

            C_t = _get_clip_value(info.get("optimizer"))
            step = int(last_step_info["step"])
            test_acc = float("nan")

            # first100 logging removed

            while milestone_idx < len(milestones):
                target = float(milestones[milestone_idx])
                if eps < target:
                    break
                _m_loss, m_acc = evaluate(
                    model=model,
                    loader=test_loader,
                    device=device,
                    criterion=criterion,
                    split_tag=f"eval_eps{float(target):.1f}",
                )
                events.append(
                    {
                        "epoch": int(epoch),
                        "step": int(step),
                        "epsilon_target": float(target),
                        "epsilon_actual": float(eps),
                        "C_t": float(C_t),
                        "test_accuracy": float(m_acc),
                        "dataset": meta_record["dataset"],
                        "method": meta_record["method"],
                        "seed": meta_record["seed"],
                    }
                )
                milestone_idx += 1

            return False

        _train_loss, _train_acc, stopped_early = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            criterion=criterion,
            epoch=epoch,
            privacy_engine=privacy_engine,
            delta=float(args.delta),
            on_batch_end=on_batch_end,
        )
        _test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
        )

        eps_end = _get_epsilon()
        C_end = _get_clip_value(optimizer)

        record = build_epoch_record(
            epoch=epoch,
            test_accuracy=test_acc,
            privacy_engine=privacy_engine,
            delta=float(args.delta),
            meta=meta_record,
            C_t=float(C_end),
        )
        records.append(record)

        write_epoch_csv(csv_path, records)
        write_epoch_json(json_path, records)

        if events:
            event_path = out_dir / f"{run_name}_events.csv"
            with event_path.open("w", newline="") as f:
                import csv

                fieldnames = [
                    "epoch",
                    "step",
                    "epsilon_target",
                    "epsilon_actual",
                    "C_t",
                    "test_accuracy",
                    "dataset",
                    "method",
                    "seed",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for ev in events:
                    writer.writerow({k: ev.get(k, "") for k in fieldnames})

        # first100 logging removed

        if scheduler is not None:
            scheduler.step()

        if stopped_early or (args.target_epsilon is not None and eps_end >= float(args.target_epsilon)):
            break

    print(f"Done. Wrote: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
