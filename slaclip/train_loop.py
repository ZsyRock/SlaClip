from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def _extract_logits(outputs) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if hasattr(outputs, "logits") and torch.is_tensor(outputs.logits):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs and torch.is_tensor(outputs["logits"]):
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def _forward_model(model: nn.Module, inputs, device: torch.device) -> torch.Tensor:
    if isinstance(inputs, dict):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        outputs = model(**inputs)
    else:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
    return _extract_logits(outputs)


def train_one_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    privacy_engine=None,
    delta: float | None = None,
    on_batch_end=None,
) -> Tuple[float, float, bool]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    stopped_early = False

    for step_idx, batch in enumerate(loader, start=1):
        inputs, targets = batch
        if targets.numel() == 0:
            continue
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_model(model, inputs, device)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = int(targets.shape[0])
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits, targets) * bs
        count += bs

        eps = float("nan")
        if privacy_engine is not None and delta is not None:
            try:
                eps = float(privacy_engine.accountant.get_epsilon(delta=float(delta)))
            except Exception:
                eps = float("nan")

        if on_batch_end is not None:
            running_acc = total_acc / max(1, count)
            should_stop = on_batch_end(
                {
                    "epoch": int(epoch),
                    "step": int(step_idx),
                    "epsilon": float(eps),
                    "batch_acc": float(_accuracy(logits, targets)),
                    "running_acc": float(running_acc),
                    "optimizer": optimizer,
                }
            )
            if should_stop:
                stopped_early = True
                break

    if count == 0:
        return 0.0, 0.0, stopped_early
    return total_loss / count, total_acc / count, stopped_early


def evaluate(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    split_tag: str = "eval",
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            if targets.numel() == 0:
                continue
            targets = targets.to(device, non_blocking=True)

            logits = _forward_model(model, inputs, device)
            loss = criterion(logits, targets)

            bs = int(targets.shape[0])
            total_loss += float(loss.item()) * bs
            total_acc += _accuracy(logits, targets) * bs
            count += bs

    if count == 0:
        result = (0.0, 0.0)
    else:
        result = (total_loss / count, total_acc / count)

    if was_training:
        model.train()

    return result


def build_epoch_record(
    *,
    epoch: int,
    test_accuracy: float,
    privacy_engine,
    delta: float,
    meta: Dict,
    C_t: float,
) -> Dict:
    if privacy_engine is None:
        epsilon = float("nan")
    else:
        try:
            epsilon = float(privacy_engine.accountant.get_epsilon(delta=float(delta)))
        except Exception:
            epsilon = float("nan")

    record = {
        "epoch": int(epoch),
        "epsilon": float(epsilon),
        "test_accuracy": float(test_accuracy),
        "C_t": float(C_t),
        "dataset": meta.get("dataset", ""),
        "method": meta.get("method", ""),
        "seed": meta.get("seed", ""),
    }
    return record
