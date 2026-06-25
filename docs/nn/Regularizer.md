# `Regularizer`

The weight-penalty interface: adds a regularization term to weight gradients during training.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules.regularizer`).

Up: [index](../_index.md).

## What it is for

A `Regularizer` discourages large weights by adding a penalty's gradient into the weight gradients each step (weight decay). Attach one (e.g. [`L2Reg`](L2Reg.md)) to a layer's weight matrix when configuring a net. Each instance carries a `strength` (the penalty coefficient, set via the constructor), and the global training loop passes a `globalRegularization` scale that is multiplied in.

## Configuration — `Regularizer(double s)`

`s` becomes the per-regularizer `strength` (stored as `float`). The effective coefficient applied is `globalRegularization * strength`.

## Driving the penalty — `regularize(weights, gradients, o, n1, s1, n2, s2, globalRegularization)`

`final` template method that walks a 2D weight block: offset `o`, with `n2` rows (stride `s2`) of `n1` weights (stride `s1`). It computes `eff = globalRegularization * strength` and calls the abstract per-row hook for each row.

## Per-row hook — `regularize(weights, gradients, o, n, s, eff)`

Abstract; subclasses add their penalty gradient for `n` weights starting at `o` with stride `s`, scaled by `eff`.

## Gotchas / dead code

- `strength` is package-private with no getter/setter — configurable only at construction.
- The two-`regularize` overload split (public 2D iterator vs. abstract 1D row) means subclasses only implement the row form.

## Cross-references

[`L2Reg`](L2Reg.md) (the concrete L2 / weight-decay penalty); `org.siquod.ml.neural1.ParamSet` (weights/gradients container).
