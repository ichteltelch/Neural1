# `SoftMaxNllLoss`

The classification loss head: log-softmax over the outputs followed by negative-log-likelihood loss.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules.loss`).

Up: [index](../_index.md).

## What it is for

The standard head for a classifier net. It takes the net's raw output logits, applies a `LogSoftmax`, and computes the NLL against a `target` distribution — producing the scalar `loss` the trainer minimizes. With [`LossGroup`](LossGroup.md)s it does this independently for several distributions packed into one output vector. Extends `LossLayer`.

## Construction — `SoftMaxNllLoss(...)`

- `SoftMaxNllLoss(String... ph)` — one softmax+NLL over the whole output (no grouping); `ph` are the training phase names in which the loss is active.
- `SoftMaxNllLoss(LossGroup[] lgs, int extraGates, String... ph)` — grouped variant: `lgs` partitions the outputs (see `LossGroup`), `extraGates` reserves additional gating inputs in the target. Internally builds a `LogSoftmax(lgs)` and an `NllLoss(lgs, extraGates, ph)`.

## Interfaces — `allocate(...)`

Reads `"in"` (logits), allocates a hidden interface named `"LogSoftMax output"` of the same shape, wires the softmax `in → hidden` and the NLL `hidden → target/loss`. It expects a `"target"` interface (the labels / target distribution) and writes a `"loss"` output. `extraGates()` forwards the loss's gate count. No learnable parameters of its own (`getParamBlocks()` returns null).

## Role in a net

Placed as the final module of a `StackModule` (typically via `addFinalLayer`). Forward runs softmax then loss; backprop runs loss then softmax, injecting the gradient back into the net. `getSubmodules()` exposes the inner softmax and NLL modules.

## Gotchas / dead code

- The class is the composition of `LogSoftmax` + `NllLoss`; configure grouping/gating through the `LossGroup[]` you pass, not on this class directly.
- `hiddenName` is the public constant `"LogSoftMax output"`.

## Cross-references

- [LossGroup](LossGroup.md) — partitions outputs into per-distribution loss groups.
- `LogSoftmax`, `NllLoss`, `LossLayer` — the underlying pieces / base class.
- [StackModule](StackModule.md) — where the head is attached.
