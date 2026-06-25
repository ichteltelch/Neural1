# `Dropout`

Regularization layer that randomly zeroes activations during training and rescales at test time.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

Standard dropout regularization. During `TRAINING` it sets a random subset of activations to zero (sampled once per run via `initializeRun`); during inference it instead multiplies every activation by `keepProbability` so expected magnitudes match. Shape-preserving (output shares the input's interface/offset). A `StackModule` adds it via a factory, since it needs to see the incoming `Interface`.

## Construction — factories and constructor

Normally added through a static `factory(...)` (an `InOutCastFactory`):

- `factory(double keepProb)` — keep-probability, new RNG, per-channel.
- `factory(double keepProb, Random rand)`
- `factory(double keepProb, boolean perChannel)`
- `factory(double keepProb, Random rand, boolean perChannel)`

`perChannel=true` drops whole channels together (uses the input `TensorFormat`); `false` drops each of the `in.count` elements independently. The raw constructor `Dropout(Interface in, double p, Random r, boolean perChannel)` is used by the factories.

## Role in a net

- `keepProbability` (public) — probability an element is kept.
- Extends `InOutCastLayer`; output reuses the input storage (an in-place cast), so it does not allocate a separate output vector.
- No learnable parameters (`getParamBlocks()` returns null).

## Gotchas / dead code

- Must **not** sit inside a convolution — `forward` throws `IllegalArgumentException` if given a position.
- The "scale by dropout mask" lines (the `a.mult(...)` / `e.mult(...)` loops) are commented out; the live path hard-zeroes instead.
- `dropoutOffset` is allocated lazily in `allocate` if still `-1`.

## Cross-references

- [StackModule](StackModule.md) — `addLayer(InOutCastFactory...)` is how `Dropout` is inserted.
- `InOutCastLayer`, `InOutCastFactory` — base class / factory type.
- [BatchNorm](BatchNorm.md) / [BatchReNorm](BatchReNorm.md) — other regularizing layers.
