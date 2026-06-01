<p align="center">
  <img src="slaclip_logo.png" width="240" alt="SlaClip logo">
</p>

# SlaClip

Official code for the ICML 2026 paper **“SlaClip: Gradient Norm Slacks can be Indicator for Adaptive Clipping in DP-SGD.”**

* **ICML Poster:** https://icml.cc/virtual/2026/poster/66390

## Overview

This repository provides a self-contained overlay for the official [Opacus](https://github.com/pytorch/opacus) repository. It adds SlaClip and the paper baselines without modifying upstream Opacus files in place. Removing the `SlaClip/` directory restores the upstream Opacus behavior.

The main components are:

* `run_exp.py`: unified experiment entry point
* `slaclip/`: argument parsing, data loading, models, training loop, and logging
* `patches/opacus/`: patched `PrivacyEngine` and optimizer implementations
* `tests/`: lightweight regression tests
* `verify_install.py`: checks that the overlay is active
* `OPACUS_BASE_VERSION.txt`: records the Opacus version or commit used by this overlay

## Quick Start

Clone the official Opacus repository and place this repository as `SlaClip/` under the Opacus root:

```bash
git clone https://github.com/pytorch/opacus.git
cd opacus
git clone https://github.com/ZsyRock/SlaClip.git SlaClip
```

Then create the environment, install Opacus in editable mode, and verify the overlay:

```bash
conda env create -f SlaClip/environment.yml
conda activate opacus
pip install -e .
python SlaClip/verify_install.py
```

To run a default SlaClip experiment on CIFAR-10:

```bash
python SlaClip/run_exp.py \
  --dataset cifar10 \
  --method slaclip \
  --seed 42 \
  --epochs 90 \
  --batch-size 2048 \
  --batch-size-test 1024 \
  --optim SGD \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --grad-sample-mode hooks \
  --delta 1e-5 \
  --lr 0.1 \
  --lr-schedule cos \
  --C0 5 \
  --sigma 1.415787 \
  --target-epsilon 8 \
  --K 50 \
  --eta 0.2 \
  --run-name cifar10_slaclip_sd42_bs2048_lr0.1_schedcos_C05_K50_eps8_sigma1.4157
```

## Requirements

Use the Opacus version or commit specified in:

```text
SlaClip/OPACUS_BASE_VERSION.txt
```

The environment file installs the required Python dependencies:

```bash
conda env create -f SlaClip/environment.yml
conda activate opacus
```

Note: `environment.yml` installs PyTorch via pip. GPU support depends on the PyTorch wheel installed on your machine. If a CPU-only build is installed, reinstall PyTorch with the appropriate CUDA wheel index, for example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## Installation Check

After installing Opacus in editable mode, run:

```bash
pip install -e .
python SlaClip/verify_install.py
```

The verification script checks whether the SlaClip overlay is active.

## Methods

Use `--method` with one of the following options:

* `slaclip`
* `slaclip-q`
* `vanilla-clip`
* `adap-clip`
* `dc-sgd-e`
* `autoclip`

## Datasets

Use `--dataset` with one of the following options:

* `mnist`
* `fmnist`
* `cifar10`
* `imdb`
* `names`

## Outputs

Outputs are written to:

```text
SlaClip/outputs/
```

Each run produces:

* `<run_name>.csv`: epoch-level results, including `epoch`, `epsilon`, `test_accuracy`, `C_t`, `dataset`, `method`, and `seed`
* `<run_name>.json`: run configuration and result summary
* `<run_name>_events.csv`: epsilon milestone events

The `_events.csv` file records the first step at which ε reaches or exceeds each target milestone. The default milestones are:

```text
0.1, 0.2, ..., 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0
```

## Interpreting Results

* `epsilon` is the privacy budget at the last batch of each epoch.
* `test_accuracy` is computed on the test set.
* `C_t` is the clipping threshold used by the method at the corresponding epoch.
* The warning `Secure RNG turned off` means that a faster non-cryptographic random number generator is used for convenience. For strict paper-grade runs, set `secure_mode=True` in `PrivacyEngine`.

## Tests

Run the lightweight regression tests with:

```bash
pytest SlaClip/tests
```

## Overlay Details

See:

```text
SlaClip/PATCH_MANIFEST.md
```

This file describes the Opacus components patched by the overlay.

## Citation

If you use this code, please cite both the SlaClip paper and Opacus.

```bibtex
@inproceedings{TODO,
  title     = {SlaClip: Gradient Norm Slacks can be Indicator for Adaptive Clipping in DP-SGD},
  author    = {TODO},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

This codebase is built as an overlay on top of the official Opacus repository. Please also cite Opacus; see the Opacus README for the official BibTeX entry.

## Notes

* This overlay relies on the official Opacus package in the parent repository.
* IMDB loading expects Hugging Face `datasets` and `transformers` support from the environment file.
* `environment.yml` installs PyTorch from pip; adjust the CUDA wheel source if your machine requires a different build.
