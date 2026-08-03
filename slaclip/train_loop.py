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
    if (
        isinstance(outputs, dict)
        and "logits" in outputs
        and torch.is_tensor(outputs["logits"])
    ):
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
    expose_training_metrics: bool = True,
    report_epsilon_on_step: bool = True,
) -> Tuple[float, float, bool, int]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    stopped_early = False
    logical_steps = 0

    def current_epsilon() -> float:
        if privacy_engine is not None and delta is not None:
            try:
                return float(privacy_engine.accountant.get_epsilon(delta=float(delta)))
            except Exception:
                pass
        return float("nan")

    def is_logical_release() -> bool:
        # BatchMemoryManager marks intermediate physical chunks as skipped.
        return not bool(getattr(optimizer, "_is_last_step_skipped", False))

    def emit_callback(
        *, physical_step: int, batch_acc: float, empty_batch: bool
    ) -> bool:
        nonlocal logical_steps
        if not is_logical_release():
            return False
        logical_steps += 1
        if on_batch_end is None:
            return False
        return bool(
            on_batch_end(
                {
                    "epoch": int(epoch),
                    "physical_step": int(physical_step),
                    "logical_step": int(logical_steps),
                    "epsilon": (
                        current_epsilon()
                        if report_epsilon_on_step
                        else float("nan")
                    ),
                    "batch_acc": (
                        float(batch_acc)
                        if expose_training_metrics
                        else float("nan")
                    ),
                    "running_acc": (
                        float(total_acc / max(1, count))
                        if expose_training_metrics and count
                        else float("nan")
                    ),
                    "empty_batch": bool(empty_batch),
                    "optimizer": optimizer,
                }
            )
        )

    for step_idx, batch in enumerate(loader, start=1):
        inputs, targets = batch
        if targets.numel() == 0:
            # A Poisson sampler can draw an empty logical batch. Skipping it
            # would silently change both the fixed-step paper mechanism and its
            # accountant. Opacus DP optimizers represent it using a leading
            # grad-sample dimension of zero, which releases pure Gaussian noise
            # and invokes the attached accountant hook exactly once. For a
            # non-private custom run, there is no mechanism to execute.
            if privacy_engine is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            parameters = getattr(optimizer, "params", None)
            if parameters is None:
                raise RuntimeError(
                    "DP optimizer does not expose parameters for an empty batch"
                )
            for parameter in parameters:
                parameter.grad_sample = torch.empty(
                    (0,) + tuple(parameter.shape),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            optimizer.step()
            should_stop = emit_callback(
                physical_step=step_idx,
                batch_acc=float("nan"),
                empty_batch=True,
            )
            optimizer.zero_grad(set_to_none=True)
            if should_stop:
                stopped_early = True
                break
            continue
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_model(model, inputs, device)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_accuracy = float("nan")
        if expose_training_metrics:
            bs = int(targets.shape[0])
            batch_accuracy = _accuracy(logits, targets)
            total_loss += float(loss.item()) * bs
            total_acc += batch_accuracy * bs
            count += bs

        should_stop = emit_callback(
            physical_step=step_idx,
            batch_acc=batch_accuracy,
            empty_batch=False,
        )
        if should_stop:
            stopped_early = True
            break

    if not expose_training_metrics:
        # Per-example training losses and labels are private queries unless a
        # separate DP measurement mechanism and budget are supplied. The paper
        # runner suppresses them for every private method.
        return float("nan"), float("nan"), stopped_early, logical_steps
    if count == 0:
        return 0.0, 0.0, stopped_early, logical_steps
    return total_loss / count, total_acc / count, stopped_early, logical_steps


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
    train_loss: float,
    train_accuracy: float,
    validation_loss: float | None,
    validation_accuracy: float | None,
    test_loss: float | None,
    test_accuracy: float | None,
    logical_steps_completed: int,
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
        "logical_steps_completed": int(logical_steps_completed),
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "validation_loss": (
            float(validation_loss) if validation_loss is not None else float("nan")
        ),
        "validation_accuracy": (
            float(validation_accuracy)
            if validation_accuracy is not None
            else float("nan")
        ),
        "test_loss": float(test_loss) if test_loss is not None else float("nan"),
        "epsilon": float(epsilon),
        "delta": float(delta),
        "test_accuracy": (
            float(test_accuracy) if test_accuracy is not None else float("nan")
        ),
        "C_t": float(C_t),
        "dataset": meta.get("dataset", ""),
        "method": meta.get("method", ""),
        "protocol": meta.get("protocol", ""),
        "phase": meta.get("phase", ""),
        "seed": meta.get("seed", ""),
    }
    return record
