# `Adam`

The Adam optimizer: adaptive per-parameter learning rates from running estimates of the gradient's first and second moments.

Source folder: `Neural1` (package `org.siquod.ml.neural1.optimizers`).

Up: [index](../_index.md).

## What it is for

`Adam` is a concrete [`Updater`](Updater.md) you select as the training optimizer. It keeps two running buffers per parameter — `m` (mean of gradients) and `v` (mean of squared gradients) — and uses them, with bias correction, to scale each parameter's step. Robust default choice for most nets.

## Configuration — public fields

- `beta1` (default `0.9f`) — decay of the first-moment (mean) estimate.
- `beta2` (default `0.999f`) — decay of the second-moment (variance) estimate.
- `epsilon` (default `1e-8f`) — denominator stabilizer.

Set these directly on the instance before/at training setup. `beta1exp`/`beta2exp` (package-private) are the running powers used for bias correction and are managed internally.

## Applying a step — `apply(ps, grad, lrMult, lr, totalWeight)`

Advances `beta1exp *= beta1` / `beta2exp *= beta2`, lazily allocates the `m`/`v` buffers on first call, then delegates the actual update to `ps.adam(lr, lrMult, beta1, beta2, epsilon, beta1exp, beta2exp, grad, m, v, totalWeight)`.

## Gotchas / dead code

- `m`/`v` are allocated on the first `apply`, sized to `ps.size()`; the same `Adam` instance must always be used with a same-sized `ParamSet`.
- `cloneData()` deep-copies `m`/`v` (null-safe).
- `toString()` exposes the betas, epsilon, and current bias-correction exponents — useful for logging optimizer state.

## Cross-references

[`Updater`](Updater.md) (superclass); [`AmsGrad`](AmsGrad.md) (variant with a max-of-`v` safeguard); `org.siquod.ml.neural1.ParamSet#adam` (the math).
