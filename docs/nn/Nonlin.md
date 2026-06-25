# `Nonlin`

Elementwise activation layer: applies one `Neuron` nonlinearity to every input element.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

The activation function in a net. It has no parameters; it simply maps each input element `x` to `n.f(x)` and produces an output of the same size. Placed after a `Dense` (or other) layer to add nonlinearity (ReLU, tanh, sigmoid, ... — whichever `Neuron` you pass).

## Construction — `Nonlin(Neuron n)`

The single constructor takes a `Neuron` (from `org.siquod.ml.neural1.neurons`) that defines `f(x)` (forward) and `df(x)` (derivative for backprop). The output interface is allocated to the same count as the input, inheriting its `TensorFormat`.

## Role in a net

Input → output is shape-preserving and elementwise. Supports the convolutional path (when given a position `pos`, operates per channel). No `ParamBlocks` (`getParamBlocks()` returns null); nothing to learn or regularize.

## Gotchas / dead code

- `copy()` returns `this` (the module is treated as stateless/shareable), not a fresh instance — fine because it holds no per-instance parameters.
- The class Javadoc says "applies the same nonlinearity to all its input elements" — the nonlinearity is fixed for the whole layer.

## Cross-references

- [ParameterizedNonlin](ParameterizedNonlin.md) — like `Nonlin` but with a learnable per-channel parameter.
- [Dense](Dense.md) — usually precedes a `Nonlin`.
- `Neuron` — defines the actual activation function.
