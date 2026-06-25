# `TransformedCursor`

A decorator cursor that applies a `CursorTransformer` to each example's inputs and/or outputs and a `DoubleUnaryOperator` to its weight, as the data streams through.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

`TransformedCursor<BT extends TrainingBatchCursor>` wraps a backing cursor and post-processes every example on the way out: after the backing cursor fills the input/output arrays, the corresponding [`CursorTransformer`](CursorTransformer.md) mutates them in place; the weight is passed through an optional `DoubleUnaryOperator`. Any of the three transforms may be `null` (pass-through). This is the implementation behind `TrainingBatchCursor.transformInputs/transformOutputs/transformWeights(...)`.

Iteration (`next`/`reset`/`isFinished`) is pure delegation — the transform is a stateless lens over the read side, so position and end-of-sequence are entirely the backing cursor's concern. A separate `TrainingBatchCursor.TransformedCursor_RA` subclass adds `size`/`seek` to preserve random access through the transform.

A transformer is allowed to *change* the vector dimension. `TransformedCursor` accounts for this at construction by sizing `inputCount`/`outputCount` to `max(backCount, transformer.dimAfterTransform(backCount))` — i.e. large enough to hold either the pre- or post-transform vector, since the transform mutates the array in place and callers must allocate the larger size.

## Construction & dimension handling — `TransformedCursor(back, inputTransformer, outputTransform, weightTransformer)`

Stores the backing cursor and the three transforms, then precomputes the reported dimensions:

- `inputCount = inputTransform==null ? back.inputCount() : max(back.inputCount(), inputTransform.dimAfterTransform(back.inputCount()))`
- `outputCount` analogously for the output side.

So a transform that grows the vector causes the cursor to advertise the larger size; one that shrinks it still advertises the original (max), since the array must hold the pre-transform data before shrinking.

## Reads — `giveInputs` / `giveOutputs` / `getWeight`

- `giveInputs(double[])` — backing cursor fills the array, then (if non-null) `inputTransform.transform(array)` mutates it in place.
- `giveOutputs(double[])` — same for the output transform.
- `getWeight()` — backing weight passed through `weightTransform.applyAsDouble(w)` if present, else returned unchanged.

## Iteration & cloning — `next` / `reset` / `isFinished` / `clone`

`next`, `reset`, `isFinished` delegate to `back`. `clone()` clones the backing cursor and re-wraps it with the same (immutable, shared) transform objects.

## Gotchas / dead code

- Transforms are applied **in place** on the caller's array, and the array must be sized to the cursor's reported `inputCount()`/`outputCount()` (the `max`), not the backing cursor's — important when a transformer expands dimensions.
- The transform objects are shared across clones (not deep-copied). Fine for stateless transformers; a stateful `CursorTransformer` would be shared between independently-iterating clones.
- No null-checks beyond the per-call `!=null` guards; a transformer that returns an inconsistent `dimAfterTransform` vs. actual `transform` length can over/under-run the array.

## Cross-references

- [`CursorTransformer`](CursorTransformer.md) — the per-example transform interface (`transform`, `dimAfterTransform`).
- [`TrainingBatchCursor`](TrainingBatchCursor.md) — defines the `transformInputs/Outputs/Weights(...)` factories that build this, plus `TransformedCursor_RA` (the random-access subclass).
- `org.siquod.ml.data.TrainingBatchCursor.WhitenedTrainingBatchCursor` — a related but specialized (linear, dimension-preserving) transform decorator.
