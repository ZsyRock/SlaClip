# SlaClip Latest

This repository is a clean extract of the functional `SlaClip/` overlay currently used in
`/home/sz1c24/opacus_01_May`. It is intended to be uploaded as a standalone GitHub repository
and then copied into an Opacus checkout for reproduction.

The repository contains only source code, tests, and environment files. It does not include
datasets, logs, or experiment outputs.

## What this overlay does

It overlays the official Opacus repository without modifying upstream files in place.
The main components are:

- `run_exp.py`: unified experiment entry point
- `slaclip/`: argument parsing, data loading, models, training loop, logging
- `patches/opacus/`: patched `PrivacyEngine` and optimizer implementations
- `tests/`: lightweight regression tests
- `verify_install.py`: checks that the overlay is active

## Expected layout

Clone Opacus, then place this repository inside the Opacus repo root as a folder named `SlaClip`.
For example:

```bash
git clone https://github.com/pytorch/opacus.git
cd opacus
cp -r /path/to/SlaClip_latest ./SlaClip
```

Using the folder name `SlaClip` is recommended because the example commands below assume that path.

The Opacus base version used for this overlay is recorded in [OPACUS_BASE_VERSION.txt](OPACUS_BASE_VERSION.txt).

## Install

From the Opacus repo root:

```bash
conda env create -f SlaClip/environment.yml
conda activate opacus
pip install -e .
python SlaClip/verify_install.py
```

`verify_install.py` checks two things:

- `opacus` resolves from your local Opacus checkout rather than `site-packages`
- patched modules resolve from `SlaClip/patches`

## Supported methods

Use `--method` with one of:

- `slaclip`
- `slaclip-q`
- `vanilla-clip`
- `adap-clip`
- `dc-sgd-e`
- `autoclip`
- `nondp`

## Supported datasets

Use `--dataset` with one of:

- `mnist`
- `fmnist`
- `cifar10`
- `imdb`
- `names`

## Example command

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
  --sigma 1.415787190199 \
  --target-epsilon 8 \
  --K 50 \
  --eta 0.2 \
  --run-name cifar10_slaclip_sd42_bs2048_lr0.1_schedcos_C05_K50_eta0.2_eps8_sigma1.415787190199
```

If `--K` is omitted for `slaclip` or `slaclip-q`, the code auto-selects the paper rule from
batch size and sigma, implemented in [slaclip/args.py](slaclip/args.py).

If `--calibrate-sigma` is used together with `--target-epsilon`, `run_exp.py` calibrates sigma
with the local Opacus accountant before training.

## Outputs

Outputs are written to `SlaClip/outputs/`:

- `<run_name>.csv`
- `<run_name>.json`
- `<run_name>_events.csv`

The epoch-level CSV/JSON contain:

- `epoch`
- `epsilon`
- `test_accuracy`
- `C_t`
- `dataset`
- `method`
- `seed`

## Patch details

See [PATCH_MANIFEST.md](PATCH_MANIFEST.md).

## Notes

- This overlay relies on the official Opacus package in the parent repository.
- IMDB loading expects Hugging Face datasets/transformers support from the environment file.
- `environment.yml` installs PyTorch from pip; adjust the CUDA wheel source if your machine needs a different build.
