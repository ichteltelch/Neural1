# `ParamSet`

The net's parameter vector: a flat `float[]` of all weights, plus the read/write and optimizer-update operations over it.

Source folder: `Neural1` (package `org.siquod.ml.neural1`).

Up: [index](../_index.md).

## What it is for

A `ParamSet` is simply a `float[] value` holding **all** of a model's learnable parameters laid out end-to-end. Each `Module` is assigned a contiguous slice (a `ParamBlock`, with a `start` offset and `count`); the module reads/writes its own weights through that block. A second `ParamSet` of the same size is used as the **gradient** buffer during training, and further same-sized sets hold optimizer state (momentum, etc.).

At the usage level you mostly:

- get a `ParamSet` from a trained model (`Eval.getParams()` / `NaiveTrainer.getParams()`),
- `clone()` it to give an `Eval` independent weights,
- `set(float[])` to load weights from disk into the right size,
- and check `size()` matches the model's `paramCount`.

The optimizer math (rprop/adam/amsgrad) lives here too but is internal — it is invoked by the `Updater`/trainer, not by application code.

## Construction & basic access — `ParamSet(int)`, `get/set/add`, `size`, `clone`

- `ParamSet(int size)` — zero-filled vector of `size` floats.
- `ParamSet(float[] v)` — copies `v` (defensive clone).
- `size()` — number of parameters; must equal the model's `paramCount`.
- `clone()` — deep copy (used to give each `Eval` its own weights).
- `clear()` / `clear(ParamBlock b)` — zero the whole vector or one block.

Element access (two flavors — by global index, or by block+local-index):

- `get(int index)` / `set(int index, double)` / `add(int index, double)` — global.
- `get(ParamBlock b, int i)` / `set(ParamBlock b, int i, double)` / `add(ParamBlock b, int i, double)` — relative to a module's block (`value[b.start + i]`); asserts the local index is in range. This is the normal way modules touch their own weights.
- `setAll(ParamBlock b, float m)` — fill one block with a constant.

## Bulk / load — `set(float[])`, `addMultiple`, `mult`, `clip`, `dot`

- `set(float[] params)` — overwrite the whole vector; throws `IllegalArgumentException("...length mismatch")` if lengths differ. This is how you load saved weights.
- `addMultiple(ParamSet ps, double factor)` — `value += ps.value * factor` (used for weighted combinations / SGD-style steps).
- `mult(float f)` — scale all values.
- `clip(float d)` — clamp every value into `[-d, d]`.
- `dot(ParamSet other)` — dot product of the two vectors (returns `double`).

## Optimizer updates — `rprop`, `adam`, `amsGrad`

Internal in-place update rules called by the trainer/`Updater`, given the gradient set, per-parameter learn-rate multipliers, and optimizer-state sets. Application code does not normally call these — it just configures an `Updater` on the `NaiveTrainer`.

- `rprop(grad, lastGrad, gamma, f, lrMult)` — resilient backprop step (per-parameter adaptive step sized in `gamma`).
- `adam(lr, lrMult, beta1, beta2, epsilon, beta1pow, beta2pow, grad, m, v, totalWeight)` — Adam.
- `amsGrad(... , vm, totalWeight)` — AMSGrad variant.

(One-line summary only; the per-element math is an optimizer detail.)

## Gotchas / dead code

- `value` is **public** and mutable — nothing stops aliasing two `ParamSet`s onto the same array, but the constructors clone, so use `clone()` rather than poking `value` directly.
- Size is everything: an `Eval` built from a wrong-size `ParamSet` throws; `set(float[])` throws on length mismatch. Always keep a `ParamSet` paired with the architecture it came from.
- All values are single-precision `float`; the `get`/`set`/`add` signatures take `double` but truncate to `float` on store.
- Several commented-out `Double.isNaN/isInfinite` guard checks remain scattered through the accessors — dead debug code.
- The bounds checks in the `ParamBlock` accessors are `assert`s, so they only fire with `-ea`.

## Cross-references

- [`FeedForward`](FeedForward.md) — `Eval`/`NaiveTrainer` carry a `ParamSet`; `paramCount` must match `size()`.
- [`Module`](Module.md) — each module owns a `ParamBlock` slice and reads/writes through it.
- `Updater` / `Rprop` (`org.siquod.ml.neural1.optimizers`), `ParamBlock` / `ParamAllocator` (`org.siquod.ml.neural1`) — drive the optimizer-update methods and block layout.
