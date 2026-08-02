import math
from types import SimpleNamespace

import torch

from slaclip.train_loop import train_one_epoch


class _Accountant:
    def __init__(self):
        self.steps = 0

    def get_epsilon(self, *, delta):
        del delta
        return float(self.steps)


class _FakeDPOptimizer:
    def __init__(self, parameters, accountant, skip_pattern=None):
        self.params = list(parameters)
        self.accountant = accountant
        self.skip_pattern = list(skip_pattern or [False])
        self._is_last_step_skipped = False
        self.calls = 0

    def zero_grad(self, set_to_none=True):
        del set_to_none
        for parameter in self.params:
            parameter.grad = None
            parameter.grad_sample = None

    def step(self):
        skip = self.skip_pattern[min(self.calls, len(self.skip_pattern) - 1)]
        self.calls += 1
        self._is_last_step_skipped = bool(skip)
        if not skip:
            self.accountant.steps += 1


def test_empty_poisson_batch_executes_one_pure_noise_accounted_release():
    model = torch.nn.Linear(2, 2)
    accountant = _Accountant()
    optimizer = _FakeDPOptimizer(model.parameters(), accountant)
    privacy_engine = SimpleNamespace(accountant=accountant)
    loader = [(torch.empty(0, 2), torch.empty(0, dtype=torch.long))]
    callbacks = []

    _, _, stopped, logical_steps = train_one_epoch(
        model=model,
        optimizer=optimizer,
        loader=loader,
        device=torch.device("cpu"),
        criterion=torch.nn.CrossEntropyLoss(),
        epoch=1,
        privacy_engine=privacy_engine,
        delta=1e-5,
        on_batch_end=lambda info: callbacks.append(info) or False,
    )

    assert stopped is False
    assert logical_steps == 1
    assert accountant.steps == 1
    assert callbacks[0]["empty_batch"] is True
    assert callbacks[0]["epsilon"] == 1.0


def test_physical_chunks_emit_callback_only_after_logical_release():
    model = torch.nn.Linear(2, 2)
    accountant = _Accountant()
    optimizer = _FakeDPOptimizer(model.parameters(), accountant, [True, False])
    privacy_engine = SimpleNamespace(accountant=accountant)
    loader = [
        (torch.ones(1, 2), torch.zeros(1, dtype=torch.long)),
        (torch.ones(1, 2), torch.zeros(1, dtype=torch.long)),
    ]
    callbacks = []

    _, _, _, logical_steps = train_one_epoch(
        model=model,
        optimizer=optimizer,
        loader=loader,
        device=torch.device("cpu"),
        criterion=torch.nn.CrossEntropyLoss(),
        epoch=1,
        privacy_engine=privacy_engine,
        delta=1e-5,
        on_batch_end=lambda info: callbacks.append(info) or False,
    )

    assert logical_steps == 1
    assert accountant.steps == 1
    assert len(callbacks) == 1
    assert callbacks[0]["physical_step"] == 2


def test_private_training_metrics_are_suppressed_from_return_and_callback():
    model = torch.nn.Linear(2, 2)
    accountant = _Accountant()
    optimizer = _FakeDPOptimizer(model.parameters(), accountant)
    privacy_engine = SimpleNamespace(accountant=accountant)
    loader = [(torch.ones(2, 2), torch.zeros(2, dtype=torch.long))]
    callbacks = []

    train_loss, train_accuracy, _, logical_steps = train_one_epoch(
        model=model,
        optimizer=optimizer,
        loader=loader,
        device=torch.device("cpu"),
        criterion=torch.nn.CrossEntropyLoss(),
        epoch=1,
        privacy_engine=privacy_engine,
        delta=1e-5,
        on_batch_end=lambda info: callbacks.append(info) or False,
        expose_training_metrics=False,
    )

    assert logical_steps == 1
    assert math.isnan(train_loss)
    assert math.isnan(train_accuracy)
    assert math.isnan(callbacks[0]["batch_acc"])
    assert math.isnan(callbacks[0]["running_acc"])
