# `L2Reg`

L2 weight regularization (weight decay): adds a penalty proportional to each weight into its gradient.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules.regularizer`).

Up: [index](../_index.md).

## What it is for

`L2Reg` is the concrete [`Regularizer`](Regularizer.md) implementing L2 weight decay — the standard "keep weights small" penalty whose gradient is proportional to the weight itself. Attach it to a layer's weights with `new L2Reg(strength)` when defining a net; larger `strength` pulls weights toward zero more strongly.

## Configuration — `L2Reg(double s)`

Passes `s` to the base [`Regularizer`](Regularizer.md) as the penalty `strength`. The effective coefficient at runtime is `globalRegularization * strength`.

## Per-row penalty — `regularize(weights, gradients, o, n, s, eff)`

For each of the `n` weights (offset `o`, stride `s`) adds `eff * weight` to the corresponding gradient — i.e. the derivative of `½·eff·w²`. `toString()` returns `"L2(strength)"`.

## Gotchas / dead code

- Pure weight decay: it modifies gradients only and never reads back the loss value, so the penalty's contribution to the reported loss is not surfaced here.

## Cross-references

[`Regularizer`](Regularizer.md) (superclass; supplies the 2D iteration and `strength`); `org.siquod.ml.neural1.ParamSet#add` (gradient accumulation).
