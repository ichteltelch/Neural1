# `Isrlu`

Inverse-Square-Root Linear Unit: a smooth, ReLU-like activation with a trainable curvature parameter `a`.

Source folder: `Neural1` (package `org.siquod.ml.neural1.neurons`).

Up: [index](../_index.md).

## What it is for

`Isrlu` is a concrete [`ParameterizedNeuron`](ParameterizedNeuron.md) you select as a layer's activation when you want a ReLU-like nonlinearity that stays smooth and saturates softly for negative inputs. For `x >= 0` it is the identity (`f = x`); for `x < 0` it bends toward a bounded floor `x / sqrt(1 + a·x²)`. The parameter `a >= 0` controls how sharply the negative side saturates (larger `a` = flatter floor); it can be trained or frozen via `fixA(a)`.

## Activation surface — `f(x,a)` / `dfdx(x,a)` / `dfda(x,a)`

- `f(x,a)` — `x` if `x>=0`, else `x/sqrt(1+a·x²)`.
- `dfdx(x,a)` — `1` on the positive branch; on the negative branch the smooth slope `(1+a·x²)^(-3/2)`.
- `dfda(x,a)` — `0` on the positive branch; a negative-branch term for learning `a`.
- `toString()` returns `"ISRLU"`.

## Gotchas / dead code

- Stateless and no public config beyond the `a` parameter passed per call — `a` is supplied by the owning layer/param set, not stored on the instance.
- Assumes `a >= 0`; negative `a` could make `1+a·x²` non-positive and break the `sqrt`.

## Cross-references

[`ParameterizedNeuron`](ParameterizedNeuron.md) (superclass; `fixA` to freeze `a`); [`Neuron`](Neuron.md) (frozen form); [`FadeInNeuron`](FadeInNeuron.md) (another parameterized activation).
