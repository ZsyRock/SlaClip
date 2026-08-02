# Patch Manifest

SlaClip prepends `SlaClip/patches` to `sys.path` so patched modules load before upstream Opacus. Upstream files are unchanged. Removing `SlaClip/` restores upstream behavior.

## Patched modules

- `opacus/__init__.py`
  - Extends package path and re-exports `PrivacyEngine`.

- `opacus/privacy_engine.py`
  - Routes paper methods and passes SlaClip parameters to the optimizer wrapper.

- `opacus/optimizers/__init__.py`
  - Extends optimizer package path and registers SlaClip/SlaClip-Q.

- Upstream `opacus/optimizers/optimizer.py`
  - Is deliberately **not** overlaid. All methods inherit the exact
    `DPOptimizer` implementation from the pinned Opacus commit.

- `opacus/optimizers/slaclipoptimizer.py`
  - Implements SlaClip with Opacus-consistent same-query release semantics under Poisson sampling.

- `opacus/optimizers/adaclipoptimizer.py`
  - Uses matched-budget accounting for the Adap-Clip baseline.

- `opacus/optimizers/autoclipoptimizer.py`
  - Implements the AutoClip baseline under the shared Opacus optimizer interface.

- `opacus/optimizers/DCSGDEOptimizer.py`
  - Implements the DC-SGD-E baseline under the shared Opacus optimizer interface.

## Guardrails

The patched `PrivacyEngine` rejects unsupported custom-method backends and
distributed modes before installing model hooks, rejects unknown optimizer
arguments in strict mode, keeps Poisson sampling aligned with the accountant,
and never reuses an explicit gradient-noise generator as a sampler generator.
Private controller checkpoints are accepted only at complete logical-query
boundaries.
