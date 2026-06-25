# `Updater`

The optimizer interface: applies a gradient step to a parameter set during training.

Source folder: `Neural1` (package `org.siquod.ml.neural1.optimizers`).

Up: [index](../_index.md).

## What it is for

`Updater` is the abstract base for gradient-descent optimizers. You pick a concrete one ([`Adam`](Adam.md) or [`AmsGrad`](AmsGrad.md)) when configuring training; the trainer calls `apply(...)` each step to move the weights along the (negated) gradient. It is `Cloneable` so per-parameter-group optimizer state can be duplicated.

## Applying a step — `apply(ParamSet ps, ParamSet grad, ParamSet lrMult, float lr, float totalWeight)`

- `ps` — the parameters to update (modified in place).
- `grad` — accumulated gradients for those parameters.
- `lrMult` — per-parameter learning-rate multipliers.
- `lr` — the base learning rate.
- `totalWeight` — normalization factor (e.g. summed sample/batch weight) the optimizer divides the gradient by.

## Cloning state — `clone()` / `cloneData()`

`clone()` shallow-copies via `Object.clone()` then calls the subclass `cloneData()` to deep-copy mutable optimizer state (moment buffers). Implementations must clone their `ParamSet` buffers there.

## Gotchas / dead code

- `clone()` swallows `CloneNotSupportedException`, prints the stack trace, and returns `null` — a failed clone yields a silent `null` rather than throwing.

## Cross-references

[`Adam`](Adam.md), [`AmsGrad`](AmsGrad.md) (concrete optimizers); `org.siquod.ml.neural1.ParamSet` (the weight/gradient/state container and the `adam`/`amsGrad` math entry points).
