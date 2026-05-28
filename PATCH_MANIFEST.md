# Patch Manifest

SlaClip prepends `SlaClip/patches` to `sys.path` so patched modules load before upstream Opacus. Upstream files are unchanged. Removing `SlaClip/` restores upstream behavior.

## Patched modules

- `opacus/__init__.py`
  - Extends package path and re-exports `PrivacyEngine`.

- `opacus/privacy_engine.py`
  - Routes paper methods and passes SlaClip parameters to the optimizer wrapper.

- `opacus/optimizers/__init__.py`
  - Extends optimizer package path and registers SlaClip/SlaClip-Q.

- `opacus/optimizers/optimizer.py`
  - Keeps the local optimizer overlay consistent with the upstream DPOptimizer interface used by the patched methods.

- `opacus/optimizers/slaclipoptimizer.py`
  - Implements SlaClip with Opacus-consistent same-query release semantics under Poisson sampling.

- `opacus/optimizers/adaclipoptimizer.py`
  - Uses matched-budget accounting for the Adap-Clip baseline.

- `opacus/optimizers/autoclipoptimizer.py`
  - Implements the AutoClip baseline under the shared Opacus optimizer interface.

- `opacus/optimizers/DCSGDEOptimizer.py`
  - Implements the DC-SGD-E baseline under the shared Opacus optimizer interface.
