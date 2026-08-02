import pytest

from opacus.accountants.utils import get_noise_multiplier


_TABLE2 = {
    "cifar10": {
        "N": 50_000,
        "epochs": 90,
        "delta": 1e-5,
        "rows": {
            512: ((4, 1.317), (6, 1.035), (8, 0.897)),
            1024: ((4, 1.739), (6, 1.317), (8, 1.109)),
            2048: ((4, 2.340), (6, 1.723), (8, 1.416)),
        },
    },
    "mnist": {
        "N": 60_000,
        "epochs": 30,
        "delta": 1e-5,
        "rows": {
            256: ((1, 1.624), (2, 1.034), (3, 0.856)),
            512: ((1, 2.189), (2, 1.309), (3, 1.031)),
            1024: ((1, 3.015), (2, 1.725), (3, 1.304)),
        },
    },
    "fmnist": {
        "N": 60_000,
        "epochs": 30,
        "delta": 1e-5,
        "rows": {
            256: ((1, 1.624), (2, 1.034), (3, 0.856)),
            512: ((1, 2.189), (2, 1.309), (3, 1.031)),
            1024: ((1, 3.015), (2, 1.725), (3, 1.304)),
        },
    },
    "imdb": {
        "N": 25_000,
        "epochs": 90,
        "delta": 1e-5,
        "rows": {
            256: ((2, 2.194), (4, 1.317), (6, 1.035)),
            512: ((2, 3.024), (4, 1.739), (6, 1.317)),
            1024: ((2, 4.176), (4, 2.340), (6, 1.723)),
        },
    },
    "names": {
        "N": 18_067,
        "epochs": 30,
        "delta": 8e-5,
        "rows": {
            256: ((1, 2.455), (2, 1.457), (3, 1.132)),
            512: ((1, 3.377), (2, 1.933), (3, 1.451)),
            1024: ((1, 4.722), (2, 2.641), (3, 1.932)),
        },
    },
}


_CASES = [
    (
        dataset,
        int(spec["N"]),
        int(spec["epochs"]),
        float(spec["delta"]),
        batch_size,
        epsilon,
        paper_sigma,
    )
    for dataset, spec in _TABLE2.items()
    for batch_size, budget_rows in spec["rows"].items()
    for epsilon, paper_sigma in budget_rows
]


@pytest.mark.parametrize(
    (
        "dataset",
        "dataset_size",
        "epochs",
        "delta",
        "batch_size",
        "epsilon",
        "paper_sigma_3dp",
    ),
    _CASES,
)
def test_all_camera_ready_table2_noise_values(
    dataset,
    dataset_size,
    epochs,
    delta,
    batch_size,
    epsilon,
    paper_sigma_3dp,
):
    del dataset
    logical_steps = -(-dataset_size // batch_size)
    sigma = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=1.0 / logical_steps,
        epochs=epochs,
        accountant="rdp",
        epsilon_tolerance=1e-5,
    )
    assert round(float(sigma), 3) == pytest.approx(paper_sigma_3dp)
