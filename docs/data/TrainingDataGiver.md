# `TrainingDataGiver`

The base interface of the cursor family: it exposes the *current* training example (input vector, output/target vector, weight) and the fixed dimensions of those vectors, without any notion of advancing.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

`TrainingDataGiver` is the read-side contract that every cursor ultimately satisfies. It describes *one* training example at a time: how many input and output variables it has, the actual values of those variables (written into caller-provided arrays), and a sample weight. It says nothing about iteration — moving to the next example and detecting the end of the sequence is added by [`TrainingBatchCursor`](TrainingBatchCursor.md), which extends this interface.

The data a giver returns is *positional*: callers learn the vector sizes from `inputCount()`/`outputCount()`, allocate arrays of that size, and pass them into `giveInputs`/`giveOutputs` to be filled. The same object can return different values over its lifetime — calling `next()`/`reset()` on the owning cursor changes what "the current example" is. A `TrainingDataGiver` on its own is therefore a snapshot view; it becomes useful when an iterating subtype drives it.

The interface also carries the `whitened(...)` decorator factory and its implementation `WhitenedTrainingDataGiver`, so any giver can be wrapped to standardize (whiten) its inputs and/or outputs.

## Vector dimensions — `inputCount()` / `outputCount()` / `usedInputCount()` / `usedOutputCount()`

- `inputCount()` / `outputCount()` (abstract) — the number of slots reserved in the input and target vectors. Callers size their arrays to these.
- `usedInputCount()` / `usedOutputCount()` (default) — by default delegate to `inputCount()`/`outputCount()`. Override them to signal that trailing indices are reserved but unused, so client code may discard them. This is purely advisory metadata; `giveInputs`/`giveOutputs` still fill the full `inputCount()`/`outputCount()` range.

## Reading the current example — `giveInputs(double[])` / `giveOutputs(double[])` / `getWeight()`

- `giveInputs(double[] inputs)` — writes the current example's input variables into the caller's array (expected length `inputCount()`). The giver does not allocate; it copies into the supplied buffer.
- `giveOutputs(double[] outputs)` — same for the target/output variables (length `outputCount()`).
- `getWeight()` — the weight of the current example, used to up-/down-weight samples during fitting.

The contract (documented on `TrainingBatchCursor`) is that these must not be called once the owning cursor `isFinished()`, until `reset()` is called.

## Whitening decorator — `whitened(Whitener, Whitener)` and `WhitenedTrainingDataGiver`

`whitened(whitenInputs, whitenOutputs)` returns a decorator that applies a `Whitener` (a linear standardizing transform) to inputs and/or outputs. Passing `null` for either skips whitening on that side.

`WhitenedTrainingDataGiver<B extends TrainingDataGiver>`:

- Validates at construction that each non-null `Whitener`'s `dim()` matches the backing giver's corresponding count, throwing `IllegalArgumentException` on mismatch.
- Allocates an internal buffer only for the side(s) actually whitened.
- `giveInputs`/`giveOutputs`: if no whitener, pass through to `back`; otherwise have `back` fill the internal buffer, then call `whitener.whiten(buffer, dest)` to write the standardized values into the caller's array.
- `inputCount`/`outputCount`/`getWeight` delegate straight to `back` — whitening preserves dimensions and weights.

`TrainingBatchCursor` overrides `whitened(...)` to return an iterating subclass (`WhitenedTrainingBatchCursor`) that extends this one.

## Gotchas / dead code

- `giveOutputs(double[] outputs)`'s Javadoc `@param` is mislabelled `inputs` (copy-paste from `giveInputs`).
- This interface has no `next()`/`reset()`/`isFinished()`; on its own it cannot be iterated. Always work through `TrainingBatchCursor` for sequences.
- `usedInputCount`/`usedOutputCount` are advisory only — nothing here enforces that the "unused" tail is actually ignored.

## Cross-references

- [`TrainingBatchCursor`](TrainingBatchCursor.md) — extends this with iteration (`next`/`isFinished`/`reset`), batching, random access, and the bulk of the factory/decorator machinery.
- `org.siquod.ml.data.Whitener` — the standardizing transform used by `whitened(...)` (not in this doc set).
- [`TransformedCursor`](TransformedCursor.md) / [`CursorTransformer`](CursorTransformer.md) — a more general (non-linear, dimension-changing) per-example transform layer.
