# `ParameterizedNeuron`

The activation-function interface with one trainable shape parameter `a` per neuron.

Source folder: `Neural1` (package `org.siquod.ml.neural1.neurons`).

Up: [index](../_index.md).

## What it is for

`ParameterizedNeuron` is the most general activation abstraction in Neural1: the nonlinearity is `f(x, a)`, where `x` is the unit's pre-activation and `a` is a per-neuron parameter the trainer can learn (e.g. the slope/curvature of [`Isrlu`](Isrlu.md), or the blend factor of [`FadeInNeuron`](FadeInNeuron.md)). A net layer picks one of these as its activation. If you do not want the parameter trained, freeze it with `fixA(a)` to obtain a plain [`Neuron`](Neuron.md).

## Activation surface — `f(x,a)` / `dfdx(x,a)` / `dfda(x,a)`

- `f(x,a)` — the activation value.
- `dfdx(x,a)` — partial derivative w.r.t. the input (for input/weight backprop).
- `dfda(x,a)` — partial derivative w.r.t. the parameter (for updating `a`).

## Freezing the parameter — `Neuron fixA(float a)`

Returns a [`Neuron`](Neuron.md) view with `a` held constant: its `f`/`df` forward to `f(x,a)`/`dfdx(x,a)`, so the parameter is no longer learned. `toString()` of the result is `"<this>(a)"`.

## Gotchas / dead code

- The `dfda` of a `fixA`-frozen neuron is effectively zero (the base `Neuron` reports `0`), so a frozen parameter contributes no gradient.

## Cross-references

[`Neuron`](Neuron.md) (non-parameterized subclass / `fixA` result); [`Isrlu`](Isrlu.md), [`FadeInNeuron`](FadeInNeuron.md) (concrete parameterized activations).
