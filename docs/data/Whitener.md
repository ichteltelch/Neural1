# `Whitener`

Interface (plus nested implementations and a static factory/linear-algebra toolbox) that standardizes feature vectors — subtracting a learned mean and multiplying by a learned (diagonal or full) inverse-standard-deviation matrix so that whitened features have ~zero mean and unit variance.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

A `Whitener` is a fitted, reusable transform applied to a feature vector before it is handed to a classifier/regressor. `whiten(in, out)` maps a raw ("colored") vector to a normalized ("whitened") one; `dim()` reports the vector length; `copy()` returns an independently-usable instance (the stateless Gaussian variants return `this`, since their parameters are immutable). The interface is `double[]`- and `float[]`-typed in parallel.

The transform is always affine: `out = invSigma · (in − µ)`. Two flavors exist:

- **`GaussianWhitener`** — diagonal: each feature is normalized independently, `out[i] = (in[i] − µ[i])·invSigma[i]`. This is per-feature standardization (z-scoring). Cheap, no decorrelation.
- **`MultivariateGaussianWhitener`** — full matrix: `out = invSigma · (in − µ)` with a dense `invSigma`, which both standardizes *and* decorrelates (rotates to the eigenbasis of the covariance and rescales). This is true whitening.

Whiteners are *fit* from data by the static `gaussianFor…`/`multivariateGaussianForInputs` factories, which stream a `TrainingBatchCursor` to accumulate weighted mean and (co)variance. They are *applied* directly via `whiten`, and (for the diagonal case) trained logistic-regression weights can be converted back and forth between the whitened and colored coordinate systems without re-fitting.

The interface body also carries a sizeable static linear-algebra library (matrix inverse, multiply, symmetric eigendecomposition via Jacobi rotations) used internally by the multivariate fit; these are general utilities that happen to live here.

## Diagonal transform — `GaussianWhitener`

Holds `double[] µ` and `double[] invSigma` (the reciprocal standard deviations). Key methods:

- `whiten(in, out)` — `out[i] = (in[i] − µ[i])·invSigma[i]`, the `float` overload casting the `double` result.
- `colorize(inWhite, outColor)` — inverse transform: `outColor[i] = inWhite[i]/invSigma[i] + µ[i]`.
- `colorizeLogarithmOfSigma(inWhite, outColor)` — `outColor[i] = inWhite[i] − log(invSigma[i])`; an additive-in-log-space variant (used when a channel carries a log-scale quantity, so dividing by sigma becomes subtracting log sigma).
- `colorizeLogisticRegressionParams(pIn, pOut)` / `whitenLogisticRegressionParams(pIn, pOut)` — convert a logistic-regression weight vector (last entry = bias) between coordinate systems so the decision function is unchanged. Derivation (in the source comment): with `whitened = diag(invSigma)·(colored − µ)`, requiring `pIn·whitened + biasIn == pOut·colored + biasOut` gives `pOut[i] = pIn[i]·invSigma[i]` and `biasOut = biasIn − Σ pIn[i]·invSigma[i]·µ[i]` (colorize), and the inverse mapping `pOut[i] = pIn[i]/invSigma[i]`, `biasOut = biasIn + Σ pIn[i]·µ[i]` (whiten). `pOut==null` allocates. This lets you train cheaply on whitened data and then deploy weights that act on raw inputs.

## Full transform — `MultivariateGaussianWhitener`

Holds `double[] µ` and `double[][] invSigma`. `whiten` computes the matrix–vector product `out[i] = Σ_j (in[j] − µ[j])·invSigma[i][j]`. No colorize/param-conversion helpers exist for this variant (it would need a matrix inverse). The `float` overload accumulates in a `float` — see Gotchas.

## Fitting the diagonal whitener — `gaussianForInputs/Outputs(…)`, `makeGaussianWhitener(…)`

Two streaming passes over the cursor:

1. Pass 1 accumulates the weighted sum `Σ w·x` and total weight `n = Σ w`, then `µ = (Σ w·x)/n`. It also tracks `all1` (whether every sample weight equals 1).
2. Pass 2 accumulates the weighted squared deviation `Σ w·(x − µ)²`, then sets `invSigma[i] = sqrt(isNorm/(sigma[i] + 1e-8))`. `isNorm` is `n − 1` when all weights are 1 (Bessel's unbiased correction) and `n` otherwise. The `+1e-8` floor prevents division by zero / infinite `invSigma` for constant features.

Variants:

- `gaussianForInputs/Outputs(TrainingBatchCursor)` — single source, on inputs vs. outputs.
- `gaussianForInputs/Outputs(TrainingBatchCursor[], ExecutorService)` — multithreaded: each cursor is summed on its own thread, partial means/variances reduced under a lock, then combined. Throws `IllegalArgumentException` on dimension mismatch.
- `gaussianForInputs/Outputs(TrainingBatchCursor, int[]... sigmaGroups)` — **grouped variance**: indices listed in the same `int[]` share a single pooled standard deviation (`sigma[group] = Σ_{i∈group} (x_i − µ_i)²`, then `invSigma[i] = sqrt(group.length · isNorm / sigma[group])`). Useful when several channels are the same physical quantity and should be scaled together. A group with zero variance yields `invSigma = 1` (left un-normalized rather than blown up); any index not assigned to a group keeps `invSigma == 0` and prints `"No variance group assigned: index i"`.

## Fitting the full whitener — `multivariateGaussianForInputs(data, regularization, preWhiten)`, `makeMultivatiateGaussianWhitener(…)`

A multi-pass procedure:

1. Compute weighted mean `µ` (pass 1).
2. Optional pre-whitening (`preWhiten`): compute per-feature `invSigma0` exactly as the diagonal whitener does (pass 2). If `preWhiten` is false, `invSigma0` is all ones. Pre-whitening makes the subsequent covariance better-conditioned when features have very different scales.
3. Accumulate the (pre-whitened) covariance `sigma[i][j] = Σ w·x̃_i·x̃_j / isNorm`, where `x̃ = (x − µ)·invSigma0`, filling only the upper triangle then mirroring (pass 3).
4. Symmetric eigendecomposition (`eigen` → `dynDiagonalize`, a cyclic Jacobi sweep, `iter=100`): `sigma = basis · diag(eigenvalues) · basisᵀ`.
5. Form `invSigma` as `basis · diag(eigenvalues^{-1/2}) · basisᵀ` (via `deDiagonalize` then `transposeInPlace`), where each eigenvalue is replaced by `1/max(sqrt(max(0, λ)), regularization)`. `regularization` floors the inverse-sqrt so near-zero eigenvalues (degenerate directions) do not explode — it is the key numerical safety knob.
6. A final correction pass (pass 4) re-measures the variance `sigma2[i]` of the data after applying `invSigma`, and rescales row `i` by `invSigma2[i]` (and column `j` by `invSigma0[j]`) so the output is exactly unit-variance per component and the pre-whitening is folded back in.

Result: `out = invSigma·(in − µ)` has approximately identity covariance.

## Linear-algebra utilities (static)

These support the multivariate fit but are general:

- `invert(dim, rows, inv)` — Gauss–Jordan inverse with partial pivoting; uses a power-of-two renormalization (`renorm = 2^{−round(log2 size)}`) during elimination to limit magnitude growth. **Mutates `rows`.**
- `mul`, `mulTranspose(dim, m1, m2, out)` (computes `m1·m2ᵀ`), `transposeInPlace`, `setUnit`, `copy`.
- `eigen(dim, of, use, basisGuess, outBasis, values, iter)` — symmetric eigendecomposition. `of` is the matrix; `use` a scratch copy (may alias `of`); `outBasis` receives eigenvectors as columns; `values` the eigenvalues. An optional `basisGuess` warm-starts via `preDiagonalize`.
- `dynDiagonalize` — the Jacobi engine: for each off-diagonal pair it solves the 2×2 eigenproblem (discriminant `detv`; skips when `detv ≤ 0`) and applies a Givens rotation (`rotL`/`rotR`) to zero the entry while accumulating the rotation into `outBasis`.
- `preDiagonalize`/`deDiagonalize` — apply `basisᵀ·M·basis` and reconstruct `basis·diag·basisᵀ`.
- `parallel`/`joinAll`/`throwCause` — small `ExecutorService` helpers; `throwCause` re-throws an `ExecutionException`'s cause if it is a `RuntimeException`/`Error`, else wraps it.

## Gotchas / dead code

- Field names use the literal Unicode `µ` (micro sign) — fine in Java identifiers but easy to mistype.
- `MultivariateGaussianWhitener.whiten(float[],float[])` accumulates the dot product in a `float` (`float sum = 0; sum += …`). For high `dim` this loses precision relative to the `double` version; the diagonal `GaussianWhitener` instead accumulates in `double` and casts once. See INCONGRUENCIES.
- `GaussianWhitener.copy()` / `MultivariateGaussianWhitener.copy()` return `this` (safe only because the parameter arrays are treated as immutable; nothing in the class mutates them, but they are `public final` references to mutable arrays).
- Two `main(…)` methods are fully commented out, plus a `mainz(…)` demo (annotated `@SuppressWarnings("unused")`) that exercises the multivariate fit on random unit-sphere polynomial features. A correctness `check` block in `makeMultivatiateGaussianWhitener` is commented out.
- `MutBool`/`MutDouble` are tiny mutable boxes used only for the lock-guarded reduction in the parallel fit.

## Cross-references

- [`RepWhitener`](RepWhitener.md) — wraps a base `Whitener` and applies it to each repetition-block of a longer interleaved vector (e.g. per-frame channels); produced by `Whitener.rep(...)`.
- `TrainingBatchCursor` / `TrainingDataGiver` — the streaming data source the factories consume (`giveInputs`/`giveOutputs`/`getWeight`/`reset`/`next`/`isFinished`).
- [`PolyInteractionCursor`](PolyInteractionCursor.md) and the `PolyInteraction*` family — a different feature-engineering transform often composed with whitening in a feature pipeline (expand features, then whiten).
- `LogisticRegression` (Neural1) — the consumer of the `…LogisticRegressionParams` conversions (referenced in the commented-out demo).
