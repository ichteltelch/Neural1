# `ParameterizedNonlin`

Elementwise activation layer with one learnable parameter per channel (e.g. PReLU-style).

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

Like [`Nonlin`](Nonlin.md), but the activation `n.f(x, alpha)` takes a learned parameter `alpha` per channel, so the network can tune the shape of its nonlinearity (e.g. the negative-slope of a parametric ReLU). Shape-preserving: output size equals input size.

## Construction — `ParameterizedNonlin(ParameterizedNeuron n, ...)`

- `ParameterizedNonlin(ParameterizedNeuron n)` — defaults `min=0`, `max=1`, `regL1=1e-6`, `regL2=1e-3`.
- `ParameterizedNonlin(ParameterizedNeuron n, float min, float max, float regL1, float regL2)` — `min`/`max` clamp the effective alpha; `regL1`/`regL2` penalize alpha drifting outside `[min,max]` (pulling it back toward the valid range).

## Parameters / init

`allocate(ParamAllocator)` allocates an `alpha` block of `in.channels()` (one per channel). `initParams` sets each alpha to the midpoint `(min+max)/2`. The `ParameterizedNeuron` provides `f(x,a)`, `dfdx` (input gradient) and `dfda` (gradient w.r.t. alpha). Alpha is bounded to `[min,max]` at use; gradient/regularization only adjust it within/towards that band.

## Gotchas / dead code

- `copy()` returns `this` (shared module). Note alpha lives in the shared `ParamSet`, so copies share parameters.
- Requires input and output `TensorFormat` to be equal (throws otherwise); it flattens to a 2D `(rows, channels)` view, so the learnable parameter is per-channel, not per-element.

## Cross-references

- [Nonlin](Nonlin.md) — the unparameterized counterpart.
- `ParameterizedNeuron` — defines `f`, `dfdx`, `dfda`.
- [Dense](Dense.md) — usually precedes the activation.
