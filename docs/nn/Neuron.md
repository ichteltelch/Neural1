# `Neuron`

The simple (non-parameterized) activation-function interface: a scalar in, a scalar out, plus its derivative.

Source folder: `Neural1` (package `org.siquod.ml.neural1.neurons`).

Up: [index](../_index.md).

## What it is for

`Neuron` is the activation a layer applies to each unit's pre-activation. It is an `abstract` subclass of [`ParameterizedNeuron`](ParameterizedNeuron.md) that drops the trainable parameter `a`: you only implement a pure function `f(x)` and its slope `df(x)`. Use a `Neuron` when the nonlinearity has no learnable shape parameter (tanh, ReLU, sigmoid, etc.). To get a `Neuron` from a parameterized one with a frozen parameter, call `parameterizedNeuron.fixA(value)`.

## Activation surface — `f(float x)` / `df(float x)`

- `f(x)` — the activation value.
- `df(x)` — its derivative `f'(x)`, used during backprop.

It wires these into the `ParameterizedNeuron` contract by making the two-arg forms final: `f(x,y)=f(x)`, `dfdx(x,y)=df(x)`, and `dfda(x,y)=0` (no parameter to train).

## Gotchas / dead code

- Subclasses override only `f(x)`/`df(x)`; the `final` two-arg overrides cannot be changed.
- The second argument of the parameterized forms is ignored — passing any `a` is harmless and has no effect.

## Cross-references

[`ParameterizedNeuron`](ParameterizedNeuron.md) (superclass; supplies `fixA` to build a `Neuron`); [`Isrlu`](Isrlu.md)/[`FadeInNeuron`](FadeInNeuron.md) (parameterized activations that can be reduced to a `Neuron`).
