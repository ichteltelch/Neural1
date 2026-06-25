# `BatchNorm`

Batch-normalization layer: normalizes each feature across the batch, then applies a learned scale and shift.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

Stabilizes/accelerates training by normalizing each output channel to zero mean and unit variance over the current batch, then rescaling with learned `mult` (scale) and `add` (shift). At test time it uses running statistics instead of the batch. Shape-preserving: input and output must have equal size (throws otherwise). Extends `BatchNormoid`.

## Construction — `BatchNorm(...)`

- `BatchNorm()` — with the learned additive shift.
- `BatchNorm(boolean hasAdd)` — `hasAdd=false` drops the `add` (bias) parameter, leaving only the scale.

## Fluent config

- `drownout(double drown)` — mixes in Gaussian noise: scales the normalized signal by `sqrt(1-drown)` and adds noise of amplitude `sqrt(drown)` (a regularizer/denoising knob; allocates a `noise` interface when nonzero).

## Parameters & statistics

`allocate(ParamAllocator)` allocates `add` (if used) + `mult`, plus `runningMean`/`runningSdev`. `initParams` sets `add=0`, `mult=1`, `runningMean=0`, `runningSdev=1`. `allocateStatistics` allocates per-batch `mean`/`var` interfaces. `updateStatistics(...)` folds batch stats into the running mean/sdev with an age-decaying weight. Accessors: `getAdd()`/`getBias()`, `getMult()`/`getScale()`, `getRunningMean()`, `getRunningSdev()`. `distributionMismatch(...)` reports how far current activations deviate from the stored distribution.

## Gotchas / dead code

- `copy()` returns `this` (shares state), unlike most modules which deep-copy.
- The convolutional (`inst != null`) path throws `UnsupportedOperationException("Not implemented")` in forward/backprop/gradients — `BatchNorm` only works outside convolutions; use [`BatchReNorm`](BatchReNorm.md) for the broadcast/conv case.
- `getParamBlocks()` mislabels itself `"Conv1D"` (copy-paste leftover).
- Test-time finalization modes (`MEANS` / `STANDARD_DEVIATIONS`) accumulate running stats during a special inference pass.

## Cross-references

- [BatchReNorm](BatchReNorm.md) — batch renormalization; supports broadcast/convolution.
- `BatchNormoid` — shared base (running-stat fields, finalization modes).
- [Dense](Dense.md) — typically normalized after.
