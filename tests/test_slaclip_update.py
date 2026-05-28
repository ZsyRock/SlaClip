import math
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_SLACLIP_DIR = _THIS_DIR.parent
_REPO_ROOT = _SLACLIP_DIR.parent
_PATCHES_DIR = _SLACLIP_DIR / "patches"

sys.path.insert(0, str(_PATCHES_DIR))
if str(_SLACLIP_DIR) not in sys.path:
    sys.path.insert(1, str(_SLACLIP_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(2, str(_REPO_ROOT))

sys.modules.pop("opacus", None)
sys.modules.pop("opacus.optimizers", None)

from opacus.optimizers.slaclipoptimizer import SlaClipOptimizer, SlaClipQOptimizer
from slaclip.args import build_parser, paper_recommended_k


def _make_slaclip_optimizer(
    *,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 2.0,
    expected_batch_size: int = 1,
    num_slots: int = 2,
    eta: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
    c_min: float = 0.1,
    c_max: float = 50.0,
):
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dp = SlaClipOptimizer(
        opt,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
        num_slots=num_slots,
        eta=eta,
        beta=beta,
        gamma=gamma,
        c_min=c_min,
        c_max=c_max,
        strict_paper_check=True,
    )
    return model, dp


def test_parser_leaves_k_unset_for_auto_selection():
    parser = build_parser()
    args = parser.parse_args(["--method", "slaclip", "--dataset", "mnist"])
    assert args.K is None


def test_paper_recommended_k_matches_table3():
    assert paper_recommended_k(128, 1.0) == 8
    assert paper_recommended_k(256, 1.0) == 10
    assert paper_recommended_k(512, 1.0) == 20
    assert paper_recommended_k(1024, 1.0) == 30
    assert paper_recommended_k(2048, 1.0) == 50


def test_slaclip_update_equation():
    _model, dp = _make_slaclip_optimizer()
    s_hat = torch.tensor([0.2, 0.4])
    C_t = 2.0
    gamma_t = max(0.0, min(1.0, 1.0 - 0.5 * (1.0 - (0.4 / 2.0))))
    expected = C_t * math.exp(0.5 * (gamma_t - 0.2))
    actual = dp._update_threshold(C_t, s_hat)
    assert abs(actual - expected) < 1e-6


def test_slaclip_q_respects_cli_bounds():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dp = SlaClipQOptimizer(
        opt,
        noise_multiplier=1.0,
        max_grad_norm=2.0,
        expected_batch_size=1,
        num_slots=2,
        eta=10.0,
        gamma=1.0,
        c_min=0.5,
        c_max=3.0,
    )
    assert dp._update_threshold(2.0, torch.tensor([0.0, 0.0])) == 3.0


def test_slack_vector_preserves_extended_norm_bound():
    _model, dp = _make_slaclip_optimizer(max_grad_norm=2.0, num_slots=4)
    C_t = 2.0
    norms = torch.tensor([0.0, 0.5, 1.25, 2.0, 3.0], dtype=torch.float32)
    slack_amount = torch.clamp(C_t - norms, min=0.0)
    lambda_t = C_t / math.sqrt(dp.K)
    slack_vector = dp._build_slack_vector(slack_amount * math.sqrt(dp.K), lambda_t)
    clipped_norms = torch.clamp(norms, max=C_t)
    extended_norms = torch.sqrt(clipped_norms.square() + slack_vector.square().sum(dim=1))
    assert torch.all(extended_norms <= (C_t + 1e-6))


def test_slack_indicator_uses_expected_batch_size_under_mean_reduction():
    model, dp = _make_slaclip_optimizer(
        noise_multiplier=0.0,
        max_grad_norm=2.0,
        expected_batch_size=8,
        num_slots=2,
    )
    param = next(model.parameters())
    param.grad_sample = torch.zeros((4,) + tuple(param.shape), dtype=param.dtype)
    param.summed_grad = torch.zeros_like(param)

    dp._slack_sum = torch.tensor([16.0, 8.0], dtype=torch.float32)
    dp._lambda_t = 2.0

    dp.add_noise()

    expected = torch.tensor([1.0, 0.5], dtype=torch.float32)
    assert torch.allclose(dp._slack_indicator, expected)


def test_slack_sum_accumulates_across_clip_calls():
    model, dp = _make_slaclip_optimizer(
        noise_multiplier=0.0,
        max_grad_norm=2.0,
        expected_batch_size=4,
        num_slots=4,
    )
    param = next(model.parameters())

    param.grad_sample = torch.tensor([[[0.5]], [[0.5]]], dtype=param.dtype)
    dp.clip_and_accumulate()

    param.grad_sample = torch.tensor([[[1.0]], [[1.0]]], dtype=param.dtype)
    dp.clip_and_accumulate()

    expected = torch.tensor([4.0, 4.0, 2.0, 0.0], dtype=torch.float32)
    assert torch.equal(dp._slack_sum, expected)
    assert dp._sample_count == 4
