# `CursorTransformer`

A tiny functional interface describing an in-place transform of an example's input or output vector, optionally changing its dimension.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

`CursorTransformer` is the per-example transform contract consumed by [`TransformedCursor`](TransformedCursor.md) (and the `transformInputs`/`transformOutputs` factories on [`TrainingBatchCursor`](TrainingBatchCursor.md)). An implementation mutates a `double[]` (the input vector or the output vector of the current example) in place, and may report a different post-transform dimension so the surrounding cursor can size its arrays accordingly.

Both methods have *default* (identity) implementations, so the interface is effectively an optional pair of hooks: a transformer that overrides only `transform` keeps the dimension; one that also overrides `dimAfterTransform` can grow or shrink the vector. Being a single-abstract... — actually it has *no* abstract methods (both are defaulted), so it is typically implemented by an anonymous/named class rather than a lambda.

## Methods — `dimAfterTransform(int)` / `transform(double[])`

- `dimAfterTransform(int dimBeforeTransform)` — returns the vector length after the transform. Default: returns the input unchanged (dimension-preserving). `TransformedCursor` uses this to compute the reported `inputCount()`/`outputCount()` as `max(before, after)`.
- `transform(double[] input)` — mutates the vector in place. Default: does nothing (identity). The array passed in is sized to the cursor's reported count (the `max` of before/after), so an expanding transformer has room to write the extra slots.

## Gotchas / dead code

- Both methods are defaulted; an empty `implements CursorTransformer {}` is a no-op identity transform.
- Because there are no abstract methods, this is **not** a functional interface usable as a lambda target — it must be implemented explicitly (this is presumably so `dimAfterTransform` and `transform` can be co-overridden).
- The parameter is named `input` even though the same interface is used for output vectors — it is just "the vector being transformed".

## Cross-references

- [`TransformedCursor`](TransformedCursor.md) — the decorator that invokes these methods and consumes `dimAfterTransform` to size its vectors.
- [`TrainingBatchCursor`](TrainingBatchCursor.md) — `transformInputs(...)` / `transformOutputs(...)` factories take `CursorTransformer`s.
