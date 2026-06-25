# `PolyInteractionCursor`

A `TrainingBatchCursor` decorator that expands each sample's input features into polynomial interactions on the fly, while passing prefix/suffix feature blocks and the outputs through untouched.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

Wraps a backing cursor `back` so that every `giveInputs(...)` call returns the polynomial-expanded feature vector instead of the raw one. This applies [`PolyInteraction`](PolyInteraction.md) feature engineering transparently inside the training-data stream, so downstream code (whitening, the classifier) sees the wider vector without anyone materializing an expanded dataset.

The raw input layout is split into three contiguous regions: a **prefix** of `prefixLength` features that pass through unchanged, the **interacting** middle block of `interactingFeatures = back.inputCount() − prefixLength − suffixLength` features that get expanded, and a **suffix** of `suffixLength` pass-through features. This lets you exclude leading/trailing channels (e.g. bias terms, raw metadata) from the polynomial blow-up.

Sizes are precomputed in the constructor:
- `interactedFeatures = simplexNumberSum(interactingFeatures, 1, order)` — the count of all monomials of degree 1…`order` over the interacting block.
- `inputCount = interactedFeatures + prefixLength + suffixLength` — the expanded width reported to consumers.
- `inSuffixOffset = prefixLength + interactingFeatures`, `outSuffixOffset = prefixLength + interactedFeatures` — where the suffix sits in the raw vs. expanded vector.

`outputCount`, `getWeight`, `reset`, `next`, `isFinished`, `giveOutputs` all delegate straight to `back`.

## Expansion — `giveInputs(double[] inputs)`

Operates **in place** on the caller's array (which must be sized `inputCount()`):
1. `back.giveInputs(inputs)` fills the raw features at the front.
2. If there is a suffix, `System.arraycopy` moves the `suffixLength` suffix features from `inSuffixOffset` up to `outSuffixOffset`, making room for the (larger) expanded block. This is done before the expansion writes over the middle, and copies forward to a higher offset.
3. `PolyInteraction.apply(interactedFeatures, 2, order, inputs, prefixLength, inputs, prefixLength + interactingFeatures)` writes the degree-2…`order` monomials. The degree-1 terms are *already present* — they are the raw interacting features sitting at `[prefixLength, prefixLength+interactingFeatures)` — so only orders ≥2 are appended after them, giving the full degree-1…`order` block in place. See INCONGRUENCIES on the first argument.

## Copy — `clone()`

Clones the backing cursor and rebuilds a `PolyInteractionCursor` with the same `order`/`prefixLength`/`suffixLength`, so the copy iterates independently (as required by `TrainingBatchCursor`).

## Gotchas / dead code

- `giveInputs` mutates the input array in place and relies on `apply` writing the order-≥2 block immediately *after* the in-place order-1 features. This works because `apply` reads its inputs (`inputs[prefixLength … +interactingFeatures]`) before/while writing strictly higher offsets; the regions do not overlap destructively.
- The first argument to `PolyInteraction.apply` is `interactedFeatures` (the expanded count) where it should arguably be `interactingFeatures` (the number of base inputs `n`). See INCONGRUENCIES.
- `giveOutputs` is overridden to delegate (`back.giveOutputs`) and `outputCount()` is overridden although the field `outputCount` is also stored — both just defer to `back`.

## Cross-references

- [`PolyInteraction`](PolyInteraction.md) — the expansion math (`apply`, `simplexNumberSum`).
- `TrainingBatchCursor` — the streaming interface being decorated.
- [`Whitener`](Whitener.md) — typically fit/applied on the expanded stream produced here.
