# `PolyInteractionFloat`

The `float`-typed twin of [`PolyInteraction`](PolyInteraction.md): expands a `float` feature vector into all monomial products of a range of orders, with matching gradient back-propagation.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

Identical purpose and algorithm to `PolyInteraction`, but operating on `float[]` arrays throughout (single-precision is enough for feature expansion fed to a classifier, and halves memory/bandwidth). Used where the surrounding pipeline already works in `float`.

## Forward expansion — `apply(…)`

Same structure as `PolyInteraction.apply`:
- Range form `apply(n, minOrder, maxOrder, in, inOffset, out, outOffset)` loops over orders and concatenates blocks, returning the total count.
- Single-order recursion: `order≤0` emits the constant `multiplier` (1); `order==1` emits `in[i]·multiplier`; otherwise recurses with `n` capped at `i+1` and `multiplier·in[i]`, which restricts factor indices to be non-increasing so each distinct monomial is emitted exactly once (combinations with repetition).

## Gradient back-propagation — `diffApply(…)`

Mirrors `PolyInteraction.diffApply` in `float`: the order-≥2 body has the two-block product-rule split (first block threads a `varIndex` for the factor being differentiated; second block continues with `multiplier·in[i]`), `order==1` accumulates into `din`, and `order≤0` throws `IllegalArgumentException`. Same offset bookkeeping; returns the count consumed.

## Gotchas / dead code

- This is essentially dead/duplicate code: `PolyInteraction` already contains a complete `float` overload of every method here, so `PolyInteractionFloat` is redundant. See INCONGRUENCIES on `PolyInteraction`.
- `simplexNumber(int,int)` is referenced in the Javadoc `@return` tags but **not defined in this class** (only `gcd` is), so those `{@link}` references dangle — the count helper lives in `PolyInteraction`.
- Carries the same `order==1`/`varIndex` `in[i]` (vs `in[inOffset+i]`) quirk and the same `main` demo as `PolyInteraction`.

## Cross-references

- [`PolyInteraction`](PolyInteraction.md) — the `double` version (and the source of the count/combinatorics helpers).
- [`PolyInteractionString`](PolyInteractionString.md) — symbolic oracle for verifying both.
- [`PolyInteractionCursor`](PolyInteractionCursor.md) — streaming application over training data.
