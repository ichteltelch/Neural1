# `BatchReNorm`

Batch-renormalization layer: batch-norm with correction factors (`r`, `d`) that bound the gap between batch and running statistics.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

A more robust variant of [`BatchNorm`](BatchNorm.md). It still normalizes per channel over the batch and applies learned `mult`/`add`, but introduces scheduled clamps `r` (ratio) and `d` (offset) that keep the batch normalization close to the running statistics, improving behaviour with small or non-i.i.d. batches. Supports a **broadcast** mode so it can normalize per channel inside convolutions (a 2D view), which `BatchNorm` cannot. Extends `BatchNormoid`.

## Construction — `BatchReNorm(...)`

- `BatchReNorm()` — `hasAdd=true`, `broadcast=true`.
- `BatchReNorm(boolean hasAdd, boolean broadcast)` — `hasAdd` toggles the learned shift; `broadcast=true` normalizes per channel over a 2D `(positions, channels)` view (suitable for convolutional inputs), `false` treats the whole input as one row.

## Fluent config

- `parallel(int threads)` — multithread the per-channel work.
- `drownout(double drown)` — noise injection, as in `BatchNorm` (`sqrt(1-drown)` signal + `sqrt(drown)` noise).
- The clamps are governed by `rmaxSchedule`/`dmaxSchedule` (functions of an `age` counter) — `r` widens from 1→3 and `d` from 0→5 over training, so early training behaves like plain batch-norm and the corrections grow as running stats stabilize.

## Parameters & statistics

`allocate(ParamAllocator)` allocates `add`(opt) + `mult` + `runningMean`/`runningSdev`, sized to `tf.channels()`. `initParams` sets `add=0`, `mult=1`, running mean/sdev to 0/1. `allocateStatistics` allocates per-batch `mean`/`sdev`; `updateStatistics` folds them into running params with age-decay. Accessors mirror `BatchNorm`: `getAdd()`/`getBias()`, `getMult()`/`getScale()`, `getRunningMean()`, `getRunningVar()` (returns the sdev block). `distributionMismatch(...)` measures deviation from stored stats.

## Gotchas / dead code

- `copy()` returns `this` (shares state).
- The convolutional `inst != null` path throws `UnsupportedOperationException("Not implemented")`; broadcast mode (not `inst`) is the supported convolution route.
- `getRunningVar()` actually returns the **sdev** param block, not a variance — misleading name (oracle: it returns `runningSdev`).

## Cross-references

- [BatchNorm](BatchNorm.md) — simpler variant without `r`/`d` correction.
- `BatchNormoid` — shared base.
- [Dropout](Dropout.md) — alternative regularizer.
