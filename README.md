# SlaClip audited reproduction overlay

Official-code reproduction support for the ICML 2026 paper **“SlaClip:
Gradient Norm Slacks can be Indicator for Adaptive Clipping in DP-SGD.”**

This directory is a versioned overlay inside an exact Opacus checkout. The
audited local layout is:

```text
Opacus-Aug/                         pinned Opacus base + packaging commit
└── SlaClip/                        independently versioned reproduction overlay
    ├── patches/opacus/             PrivacyEngine and paper optimizers
    ├── slaclip/                    protocol, data, models, loop, logging
    ├── tools/                      dry-run grid generation and selection
    ├── tests/                      CPU regression/integration tests
    ├── PAPER_FIDELITY.md           equation/guardrail/unknown audit
    └── REPRODUCIBILITY_VERSIONS.json
```

No dataset or training run is started by installation or verification.

## Exact source versions

The required Opacus base is:

```text
f17f254ab8f1f1095e8257bf278769d549748bbc
v1.5.4-21-gf17f254
```

The audited SlaClip branch is based on upstream commit:

```text
d48b8e07aef33c58a3595ee18b4dccf9c75fa1f3
```

The local reproduction commit and tags are recorded after audit in Git and in
`REPRODUCIBILITY_VERSIONS.json`. They are local until explicitly pushed by the
repository owner.

To reconstruct the parent before applying the audited overlay:

```bash
git clone https://github.com/meta-pytorch/opacus.git Opacus-Aug
cd Opacus-Aug
git checkout f17f254ab8f1f1095e8257bf278769d549748bbc
git clone https://github.com/ZsyRock/SlaClip.git SlaClip
cd SlaClip
git checkout d48b8e07aef33c58a3595ee18b4dccf9c75fa1f3
```

The local `Opacus-Aug` checkout already contains the audited overlay; do not
repeat those clone commands there.

## Environment and verification

The pinned environment targets Python 3.10, PyTorch 2.10/CUDA 12.8, and
torchvision 0.25:

```bash
cd Opacus-Aug
conda env create -f SlaClip/environment.yml
conda activate opacus
python -m pip install --no-deps -e .
python SlaClip/verify_install.py
python -m pytest -q SlaClip/tests
```

`verify_install.py` checks both overlay resolution and that the parent history
descends from the pinned Opacus base without changes to upstream Opacus code.

For an existing compatible environment, a small virtual environment using its
site packages is sufficient; avoid reinstalling multi-gigabyte CUDA wheels:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python SlaClip/verify_install.py
```

## Paper protocols

The runner fails on unknown arguments and distinguishes two protocols that use
different privacy budgets.

Controlled Appendix-F example (generates no work until this command is run):

```bash
python SlaClip/run_exp.py \
  --method slaclip \
  --dataset cifar10 \
  --protocol controlled \
  --budget-index 1 \
  --seed 42 \
  --device cuda \
  --run-name controlled-cifar10-slaclip-b1-s42
```

This fixes `sigma=1`, the controlled optimizer recipe, `C0=1`, SlaClip
`eta=0.5`, and stops after the requested logical DP step reaches the relevant
Appendix budget. For `sigma=1`, automatic K uses the practical values in
Appendix D/Table 3.

A single main-protocol selection candidate is:

```bash
python SlaClip/run_exp.py \
  --method slaclip \
  --dataset cifar10 \
  --protocol main \
  --phase selection \
  --budget-index 1 \
  --batch-size 512 \
  --lr 0.1 \
  --C0 1 \
  --lr-schedule cos \
  --acknowledge-public-validation \
  --device cuda \
  --run-name main-select-slaclip-cifar10-example
```

Main runs calibrate sigma over the full prescribed horizon with tolerance
`1e-5` (matching all 45 entries in Table 2 to three decimals). K is then chosen from Eq. (14)
using the final sigma and actual logical release denominator. Do not pass
`--sigma` or `--K` in the main protocol.

The runner freezes the paper's integer mechanism before Opacus conversion:
`q_eff = 1 / ceil(N/B)` and `T = ceil(N/B) * epochs`.  This avoids floating-
point reciprocal truncation (for example, `1 / (1/211)` falling just below
211) and asserts that the Poisson sampler, optimizer normalization, metadata,
and accountant all use the same values before training starts.  Noise
calibration receives the exact integer `T`, never a floating `epochs/q`
conversion.

A precomputed privacy guard prevents an over-budget Gaussian release. Normal
paper runs must complete all `T` releases on the safe side of the target. If
numerical calibration makes only release `T` exceed the target, the runner may
stop at `T-1` and records both the last compliant epsilon and the projected
epsilon of the omitted release. Validators accept this fallback only when
exactly one final release is omitted and `epsilon(T-1) <= target < epsilon(T)`;
larger mismatches remain hard failures.

After validation selection, retrain the chosen candidate separately with seeds
42, 43, and 44 using `--phase retrain`. Retraining restores the complete
official training split and touches the test set only at the end.

## Published shared grid

The paper publishes a shared grid of 16,200 candidates: six methods, five
datasets, three budgets, six learning rates, three dataset-specific batch
sizes, five C0 values, and two schedules. Generate manifests and shell commands
without executing any experiment:

```bash
python SlaClip/tools/generate_main_grid.py \
  --output SlaClip/outputs/main_candidates.jsonl \
  --commands-output SlaClip/outputs/main_commands.sh
```

After all candidate JSON files exist under
`SlaClip/outputs/main_selection/`, validate the complete result set, select only
by final validation accuracy, and generate 270 retrain commands (90
method/dataset/budget winners, each retrained with three seeds):

```bash
python SlaClip/tools/select_main_grid.py \
  --candidates SlaClip/outputs/main_candidates.jsonl \
  --runs-dir SlaClip/outputs/main_selection \
  --output SlaClip/outputs/main_retrains.jsonl \
  --commands-output SlaClip/outputs/main_retrains.sh
```

Both tools require clean, matching Git identities. Because SlaClip is an
independently versioned nested repository, the outer Opacus cleanliness check
ignores only the SlaClip gitlink while the exact clean SlaClip commit is checked
separately. The selector rejects missing
candidates, mismatched configurations, dirty runs, incomplete horizons, and any
selection-time test metric. The paper does not publish every method-specific
adaptive-parameter range or the authors' selected configurations; those are
recorded as underspecified rather than guessed.

## 8 GB GPU and storage safeguards

Private runs default to physical microbatches of 64 for vision and 32 for text.
`BatchMemoryManager` accumulates them into the requested logical Poisson batch,
so noise and the accountant advance once per logical batch. Reduce
`--max-physical-batch-size` further if an 8 GB GPU still approaches OOM; this
does not change logical B or q.

Datasets, tokenized IMDB caches, environments, and outputs are Git-ignored.
Names archive download/extraction has explicit compressed/uncompressed size,
member-count, symlink, and path-traversal guards. Output files are never
overwritten unless `--overwrite` is explicit. Check free space before running
the full 16,200-candidate grid; the tools themselves only write small manifests.

## Outputs and privacy boundary

Each run writes exactly:

```text
<run_name>.csv
<run_name>.json
<run_name>_config.json
```

The records include arguments, protocol and split conventions, effective
sampling rate, expected logical batch, sigma/K provenance, dependencies,
device, both Git states, epsilon, C trajectory, and C-boundary hits.

For private methods, raw training loss and accuracy are intentionally null/NaN:
publishing them would be an additional unaccounted private query. Selection
validation is an explicitly acknowledged public benchmark holdout; its 10%
ratio and seed 2026 are deterministic conventions because the paper does not
publish them. The per-command accountant does not compose privacy across a
hyperparameter sweep.

`secure_mode=False` matches the camera-ready numerical setting but is not a
cryptographic deployment guarantee. A model intended to protect genuinely
sensitive data must be retrained from the beginning with `--secure-mode` and a
working `torchcsprng`; never silently fall back.

## Fidelity status

See `PAPER_FIDELITY.md` for the equation-exact (E), equivalent (Q), declared
guardrail (G), and underspecified (U) matrix. The most important declared
engineering bounds are C in `[0.1,20]` and configurable probabilities in
`[0.01,0.99]`; SlaClip Appendix C's projection remains `[0,1]`. Adap-Clip's
noisy-fraction projection is recorded separately as a DP-safe engineering
guardrail. Guardrails are not silently treated as paper equations, and boundary
activation is logged because it can affect utility.

## Citation

```bibtex
@inproceedings{zou2026slaclip,
  title     = {SlaClip: Gradient Norm Slacks can be Indicator for Adaptive Clipping in DP-SGD},
  author    = {Zou, Shuyan and Wang, Shaowei and Zhu, Zhanxing and Li, Jin and Dong, Changyu and Sassone, Vladimiro and Wu, Han},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```

Please also cite Opacus and the original baseline papers when reporting those
methods.
