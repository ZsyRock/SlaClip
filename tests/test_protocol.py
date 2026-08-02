import math

import pytest

from slaclip.args import (
    build_parser,
    paper_k_upper_bound,
    paper_recommended_k,
    parse_args,
)
from slaclip.protocol import apply_protocol


def _parse(*extra):
    return parse_args(["--method", "slaclip", "--dataset", "cifar10", *extra])


def test_unknown_arguments_fail_loudly():
    with pytest.raises(SystemExit):
        parse_args(["--method", "slaclip", "--dataset", "mnist", "--epohs", "30"])


@pytest.mark.parametrize(
    ("expected_batch_size", "sigma"),
    [(128, 1.0), (512, 1.0), (512, 1.7), (2048, 0.9)],
)
def test_automatic_k_is_eq14_floor(expected_batch_size, sigma):
    upper = (expected_batch_size / (2 * 2.5758293035489004 * sigma)) ** (2 / 3)
    assert paper_k_upper_bound(expected_batch_size, sigma) == pytest.approx(upper)
    assert paper_recommended_k(expected_batch_size, sigma) == math.floor(upper)


def test_automatic_k_depends_on_final_sigma():
    assert paper_recommended_k(512, 0.8) > paper_recommended_k(512, 2.0)


def test_automatic_k_rejects_when_no_positive_integer_is_admissible():
    with pytest.raises(ValueError, match="no admissible positive integer K"):
        paper_recommended_k(1, 100.0)


def test_explicit_k_is_preserved_for_paper_ablation():
    args = _parse(
        "--protocol",
        "controlled",
        "--budget-index",
        "1",
        "--seed",
        "42",
        "--K",
        "100",
    )
    apply_protocol(args)
    assert args.K == 100


def test_main_protocol_rejects_explicit_k_and_uses_eq14_path():
    args = _parse(
        "--protocol",
        "main",
        "--phase",
        "selection",
        "--budget-index",
        "1",
        "--batch-size",
        "512",
        "--lr",
        "0.1",
        "--C0",
        "1",
        "--lr-schedule",
        "cos",
        "--K",
        "50",
        "--acknowledge-public-validation",
    )
    with pytest.raises(ValueError, match=r"Eq\. \(14\)"):
        apply_protocol(args)


def test_main_budget_and_full_horizon_calibration_are_unambiguous():
    args = _parse(
        "--protocol",
        "main",
        "--phase",
        "selection",
        "--budget-index",
        "2",
        "--batch-size",
        "512",
        "--lr",
        "0.1",
        "--C0",
        "1",
        "--lr-schedule",
        "cos",
        "--acknowledge-public-validation",
    )
    metadata = apply_protocol(args)
    assert args.target_epsilon == 6
    assert args.epochs == 90
    assert args.epsilon_mode == "calibrate"
    assert args.epsilon_tolerance == pytest.approx(1e-5)
    assert args.sigma is None
    assert args.seed == args.selection_seed == 2026
    assert metadata["validation_convention"]["paper_specified"] is False


def test_main_selection_requires_public_holdout_acknowledgement():
    args = _parse(
        "--protocol",
        "main",
        "--phase",
        "selection",
        "--budget-index",
        "1",
        "--batch-size",
        "512",
        "--lr",
        "0.1",
        "--C0",
        "1",
        "--lr-schedule",
        "cos",
    )
    with pytest.raises(ValueError, match="acknowledge-public-validation"):
        apply_protocol(args)


def test_main_retrain_requires_published_three_seeds():
    args = _parse(
        "--protocol",
        "main",
        "--phase",
        "retrain",
        "--budget-index",
        "1",
        "--batch-size",
        "512",
        "--lr",
        "0.1",
        "--C0",
        "1",
        "--lr-schedule",
        "cos",
        "--seed",
        "41",
    )
    with pytest.raises(ValueError, match="42, 43, 44"):
        apply_protocol(args)


def test_controlled_protocol_uses_appendix_budgets_and_fixed_sigma():
    args = _parse("--protocol", "controlled", "--budget-index", "3", "--seed", "42")
    apply_protocol(args)
    assert args.target_epsilon == 9
    assert args.sigma == 1
    assert args.epsilon_mode == "hard-stop"
    assert args.batch_size == 1024
    assert args.C0 == 1
    assert args.eta == 0.5


def test_controlled_protocol_requires_explicit_paper_seed():
    args = _parse("--protocol", "controlled", "--budget-index", "1")
    with pytest.raises(ValueError, match="explicit --seed"):
        apply_protocol(args)


def test_paper_protocol_requires_fixed_guardrails_and_gamma():
    guardrail_args = _parse(
        "--protocol",
        "controlled",
        "--budget-index",
        "1",
        "--seed",
        "42",
        "--c-max",
        "19",
    )
    with pytest.raises(ValueError, match="guardrails"):
        apply_protocol(guardrail_args)

    gamma_args = _parse(
        "--protocol",
        "controlled",
        "--budget-index",
        "1",
        "--seed",
        "42",
        "--gamma",
        "0.6",
    )
    with pytest.raises(ValueError, match="gamma=0.5"):
        apply_protocol(gamma_args)


@pytest.mark.parametrize(
    ("dataset", "main", "controlled"),
    [
        ("cifar10", (4, 6, 8), (5, 7, 9)),
        ("mnist", (1, 2, 3), (1, 2, 3)),
        ("fmnist", (1, 2, 3), (1, 2, 3)),
        ("imdb", (2, 4, 6), (2, 4, 6)),
        ("names", (1, 2, 3), (2, 4, 5)),
    ],
)
def test_main_and_controlled_budget_families_are_not_conflated(
    dataset, main, controlled
):
    from slaclip.protocol import CONTROLLED_BUDGETS, MAIN_BUDGETS

    assert MAIN_BUDGETS[dataset] == main
    assert CONTROLLED_BUDGETS[dataset] == controlled


def test_main_autoclip_uses_shared_c0_pool_not_controlled_c0_rule():
    args = parse_args(
        [
            "--method",
            "autoclip",
            "--dataset",
            "mnist",
            "--protocol",
            "main",
            "--phase",
            "retrain",
            "--budget-index",
            "1",
            "--batch-size",
            "256",
            "--lr",
            "0.1",
            "--C0",
            "5",
            "--lr-schedule",
            "constant",
            "--seed",
            "42",
        ]
    )
    apply_protocol(args)
    assert args.C0 == 5


def test_fraction_arguments_are_bounded():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--method", "adap-clip", "--dataset", "mnist", "--gamma", "1.0"]
        )


def test_custom_target_epsilon_requires_explicit_semantics():
    args = _parse("--target-epsilon", "4")
    with pytest.raises(ValueError, match="no implicit meaning"):
        apply_protocol(args)


def test_main_rejects_looser_noise_calibration_tolerance():
    args = _parse(
        "--protocol",
        "main",
        "--phase",
        "selection",
        "--budget-index",
        "1",
        "--batch-size",
        "512",
        "--lr",
        "0.1",
        "--C0",
        "1",
        "--lr-schedule",
        "cos",
        "--epsilon-tolerance",
        "0.01",
        "--acknowledge-public-validation",
    )
    with pytest.raises(ValueError, match="Table 2 noise calibration"):
        apply_protocol(args)


def test_custom_nondp_does_not_apply_private_microbatch_manager():
    args = parse_args(["--method", "nondp", "--dataset", "mnist"])
    apply_protocol(args)
    assert args.max_physical_batch_size == args.batch_size
