# Paper-fidelity audit

This file records how the implementation is mapped to the 22-page ICML 2026
camera-ready paper identified in `REPRODUCIBILITY_VERSIONS.json`. It is an
audit record, not a claim that unpublished author choices can be reconstructed.

## Status rubric

- **E — equation exact:** the public paper states the operation and a focused
  test checks the same formula or invariant.
- **Q — equivalent:** the implementation is algebraically equivalent while
  avoiding an unnecessary materialization or preserving logical batches through
  physical microbatches.
- **G — guardrail:** a declared deterministic post-processing or input bound;
  it does not change privacy accounting, but activation can change utility and
  must be logged.
- **U — underspecified:** the paper does not publish enough information to
  reconstruct the authors' exact choice. The repository uses an explicit,
  deterministic convention and labels it as such.

## SlaClip mechanism

| Claim | Status | Evidence/decision |
|---|---:|---|
| Global per-example gradient clipping at `C_t` | E | Paper Eq. (6); optimizer norm-bound tests. |
| `lambda = C_t / sqrt(K)` and prefix slack encoding | E | Paper Eq. (6)–(8); extended-norm tests. |
| One joint `d+K` Gaussian release | E | A single noise tensor is split into gradient and slack coordinates; no second slack query is callable. |
| Fixed public expected-batch normalization | E | Gradient and slack use the same Opacus logical-batch denominator. |
| First `d` coordinates preserve vanilla DP-SGD | E/Q | The paper query and Gaussian marginal are exact. The SlaClip path uses a `1e-12` denominator guard while pinned Opacus uses `1e-6`, so this is not claimed to be bitwise identical at the clipping boundary. Both remain norm-bounded by `C_t`. |
| `s_hat = released_slack / lambda` | E | Paper Eq. (10)–(11). |
| Full controller | E | Appendix C Eq. (28)–(30): `r=clip(s_hat[-1]/C_t,0,1)`, `gamma=(1+r)/2`, then the exponential update. |
| SlaClip-Q controller | E | Paper Eq. (12), with the published median target by default. |
| Automatic `K` | E | Main runs evaluate Eq. (14) from the actual logical release denominator and final calibrated sigma, then floor. Controlled `sigma=1` runs use Appendix D/Table 3 practical values (8/10/20/30/50); explicit `K` is reserved for controlled/custom ablations. |
| Poisson sampling and RDP accounting | E | Same sampling event, total noise multiplier, and accountant hook as the pinned Opacus base. |
| Physical microbatching | Q | Physical chunks are accumulated into one logical query; noise and accountant advance only once per logical batch. |
| `C_t` range `[0.1,20]` | G | User-approved stability range. Boundary hits are recorded because frequent activation changes utility. |

### Controller ambiguity resolution

The main-text compact formula/Algorithm 1 and Appendix C differ for a negative
noisy `s_hat[-1]/C_t`. Appendix C explicitly derives an intermediate
`r=clip(...,0,1)` in Eq. (28) and says that this is the rule used by SlaClip.
This repository treats the expanded Appendix C derivation as authoritative.
Both interpretations are DP-safe post-processing, but they are not utility
equivalent for negative noise; the selected interpretation is therefore tested
and recorded rather than hidden as an engineering change.

## Baselines

| Method | Status | Scope and caveat |
|---|---:|---|
| Vanilla DP-SGD | E | Standard global clipping, Gaussian sum noise, fixed expected-batch normalization. |
| Adap-Clip | E/Q/G | Published quantile controller and split privacy budget, applied here to centralized example gradients although the original work evaluated federated user updates. The sensitivity-`1/2` centered bit `1{||g_i||<=C}-1/2` with a fixed public `expected_batch_size*accumulated_iterations` denominator is algebraically equivalent to the original fixed-size count and makes the add/remove Poisson implementation auditable. Empty batches still release count and gradient noise. Projecting the resulting noisy fraction to `[0,1]` is a DP-safe engineering guardrail used by this reproduction/original repository, not an equation claimed from Andrew et al. Algorithm 1. |
| AutoClip AUTO-S | E | `R*g/(||g||+0.01)` without an Abadi clamp. Main Table 1 uses the shared `C0` pool; only the controlled Table 5 recipe fixes `R=C0=1`. |
| DC-SGD-E | E/G | Published noisy histogram, budget split, 20-point error grid, and range adaptation. An empty Poisson batch releases an all-zero histogram plus Gaussian bin noise instead of skipping the accounted mechanism. The finite boundary-search cap is a liveness guardrail and emits a warning if reached. |

User-configurable probability/quantile parameters are limited to `[0.01,0.99]`.
This does **not** replace SlaClip Appendix C's `[0,1]` projection. Adap-Clip's
noisy-fraction projection is separately classified as a **G** choice above.

These ratings cover the baseline mechanisms as instantiated by the SlaClip
camera-ready comparison. They do not claim to recreate every experimental
condition of each baseline's original paper: Adap-Clip was originally motivated
and evaluated for federated/user-level updates, while this benchmark applies its
accounted controller to centralized example gradients; AutoClip's AUTO-S
transformation is exact here, but no external pretraining pipeline from the
original AutoClip study is reconstructed. Those differences are method-scope
caveats, not ignorable implementation optimizations, and must accompany any
cross-paper claim.

## Experimental protocol

The code distinguishes the paper's two protocols instead of mixing their
budgets:

1. **Main fairly tuned comparison:** shared learning-rate, batch-size, `C0`,
   and schedule grids; budgets from Table 1; dataset-specific 30/90-epoch
   horizons; per-candidate noise calibration; selection followed by final seeds
   42/43/44.
2. **Controlled fixed recipe:** Appendix A/F fixed optimizer settings,
   `sigma=1`, `C0=1` unless explicitly ablated, and its separate diagnostic
   privacy milestones.

The paper says that one selection seed and validation accuracy were used, but
does not publish the validation source, holdout ratio, selection-seed value, the
complete selected-configuration manifest, or the ranges for all
method-specific adaptive-parameter sweeps. Those items are **U**, not E. The
implementation uses and logs a 10% stratified public holdout and selection seed
2026, never creates a test loader during selection, and restores the full
published training corpus for seeds 42/43/44. The companion grid tools enumerate
all 16,200 candidates in the *published shared* grid and refuse incomplete,
dirty, or test-contaminated result sets; they do not pretend to reconstruct
unpublished adaptive-parameter grids.

The paper/released repository also does not pin a Hugging Face revision for
IMDB or an archive digest for Names. These remain **U** and prevent a claim of
data-level bitwise reproduction. Every materialized run records the IMDB
tokenized split fingerprints and tokenizer-vocabulary SHA-256, or the per-file
and aggregate SHA-256 values for the extracted Names corpus, so later runs can
detect a data-source change instead of silently mixing versions.

## Operational DP boundary

`secure_mode=False` is retained as an explicit paper-reproduction option. It is
not a cryptographic deployment guarantee. Any model released as protecting
real sensitive data must be retrained from the beginning with secure sampling
and noise generation. Raw private training loss/accuracy is suppressed from
callbacks and output files; validation/test metrics are computed only on the
explicitly public benchmark splits. The SlaClip, Adap-Clip, and DC-SGD-E custom
controllers accept checkpoints only at complete logical-step boundaries so that
raw physical-batch counts, norms, or slack buffers are never serialized;
vanilla DP-SGD and AutoClip retain the pinned upstream checkpoint behavior.

## Engineering deviations that cannot be ignored silently

The `[0.1,20]` clipping range, `[0.01,0.99]` configurable probability range,
finite DC-SGD-E boundary-search cap, padding-aware text pooling, and physical
microbatch caps are declared **G/Q** choices. They preserve the stated privacy
accounting, but a clipping-boundary hit or a controller-search cap can change
utility. Runs therefore record boundary activation, full arguments, data split
hashes, dependencies, and both Git identities. Such changes are acceptable for
an engineering reproduction only when reported; they must not be presented as
unqualified equation-exact results.
