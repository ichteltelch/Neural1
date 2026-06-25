# `FadeInNeuron`

A wrapper activation that blends linearly between the identity and an inner activation, controlled by a trainable factor `a`.

Source folder: `Neural1` (package `org.siquod.ml.neural1.neurons`).

Up: [index](../_index.md).

## What it is for

`FadeInNeuron` lets a net "fade in" a nonlinearity during training. Construct it with `new FadeInNeuron(back)` around any inner [`Neuron`](Neuron.md) (`back`). Its output is `a·back.f(x) + (1-a)·x`: with `a=0` the unit is purely linear (identity), and as the trained parameter `a` rises toward 1 the inner activation takes over. Use it when starting training with a near-linear net and gradually introducing the nonlinearity improves convergence.

## Construction — `FadeInNeuron(Neuron back)`

`back` (public final field) is the activation that is faded in. The blend factor `a` is the parameterized neuron's trainable parameter, supplied per call.

## Activation surface — `f(x,a)` / `dfdx(x,a)` / `dfda(x,a)`

- `f(x,a)` — `a·back.f(x) + (1-a)·x`.
- `dfdx(x,a)` — `a·back.df(x) + (1-a)`.
- `dfda(x,a)` — `back.f(x) - x` (gradient that drives the fade-in).

## Gotchas / dead code

- `a` is meaningful in `[0,1]`; nothing clamps it, so an unconstrained trainer could push it outside that range (extrapolating the blend).
- No `toString()` override (unlike `Isrlu`), so it prints the default object identity.

## Cross-references

[`Neuron`](Neuron.md) (the wrapped `back` activation); [`ParameterizedNeuron`](ParameterizedNeuron.md) (superclass; `fixA` to freeze the blend); [`Isrlu`](Isrlu.md) (a candidate `back`).
