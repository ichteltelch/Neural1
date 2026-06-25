# `AmsGrad`

The AMSGrad optimizer: Adam with a running maximum of the second-moment estimate for more stable convergence.

Source folder: `Neural1` (package `org.siquod.ml.neural1.optimizers`).

Up: [index](../_index.md).

## What it is for

`AmsGrad` is a concrete [`Updater`](Updater.md), a variant of [`Adam`](Adam.md). In addition to the first/second moment buffers `m`/`v`, it keeps `vm`, the element-wise running maximum of `v`, and uses `vm` (instead of `v`) to scale steps. This prevents the effective learning rate from growing, addressing a known Adam non-convergence case. Select it when Adam is unstable.

## Configuration — public fields

- `beta1` (default `0.9f`) — first-moment decay.
- `beta2` (default `0.999f`) — second-moment decay.
- `epsilon` (default `1e-8f`) — denominator stabilizer.

`beta1pow`/`beta2pow` (package-private) are the bias-correction running powers, managed internally.

## Applying a step — `apply(ps, grad, lrMult, lr, totalWeight)`

Advances the bias-correction powers, lazily allocates `m`/`v` and `vm`, then delegates to `ps.amsGrad(lr, lrMult, beta1, beta2, epsilon, beta1pow, beta2pow, grad, m, v, vm, totalWeight)`.

## Resetting the max — `forgetVMax()`

Sets `vm = null`, dropping the accumulated second-moment maximum (re-allocated on the next `apply`). Use to "release the brake" — e.g. after a learning-rate change or schedule restart.

## Gotchas / dead code

- `cloneData()` guards every clone on `if(m!=null)` — so `v` and `vm` are cloned only when `m` is non-null. This is almost always equivalent (all three allocate together), but a state where `m==null` while `vm!=null` would silently skip cloning `vm`. See INCONGRUENCIES.
- No `toString()` override (unlike `Adam`).

## Cross-references

[`Updater`](Updater.md) (superclass); [`Adam`](Adam.md) (base algorithm); `org.siquod.ml.neural1.ParamSet#amsGrad` (the math).
