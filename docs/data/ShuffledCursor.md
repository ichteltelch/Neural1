# `ShuffledCursor`

A `RandomAccess` decorator that presents a backing cursor's items in a random order via an index permutation, without moving or copying the underlying data.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

`ShuffledCursor` wraps any [`TrainingBatchCursor.RandomAccess`](TrainingBatchCursor.md) and serves its examples in a shuffled order. It does this by maintaining an `int[] permutation` (a shuffled list of the backing indices) plus a current `index` into that permutation; iterating just `seek`s the backing cursor to `permutation[index]`. The backing data is never touched or reordered — the shuffle is entirely in the index layer. This is the per-epoch-shuffle building block of the pipeline: typically you `ramBuffer()` a source, wrap it in a `ShuffledCursor`, and call `shuffle()` between epochs.

Contrast with `TrainingBatchCursor.RamBuffer.shuffle(Random)`, which physically permutes the buffer's arrays (and all its clones). `ShuffledCursor` instead leaves the backing store fixed and reshuffles its own index array, so it can shuffle *any* `RandomAccess`, not just a `RamBuffer`.

Because the permutation is materialized as an `int[]`, the sequence length must fit in an `int`: `initPermutation()` throws `IllegalStateException` if `back.size() > Integer.MAX_VALUE`.

It carries its own `Random` (cloned from the supplied one via `DataManagement.cloneRandom`, so the caller's RNG isn't shared/advanced) and a `fresh` flag used to avoid a redundant first shuffle.

## Construction & permutation init — constructors / `initPermutation()` / `shuffle(int[])`

- `ShuffledCursor(RandomAccess back, Random rand)` — clones `rand`, then `initPermutation()`.
- `ShuffledCursor(RandomAccess back)` — uses a fresh `new Random()`.
- `initPermutation()` — guards the `Integer.MAX_VALUE` length limit, builds the identity permutation `[0,1,…,size-1]`, Fisher–Yates–shuffles it, and seeks the backing cursor to the first permuted index.
- `shuffle(int[] data)` — in-place Fisher–Yates over the given array using the internal `rand`.
- private `ShuffledCursor(back, permutation, index, rand)` — the clone constructor (does not re-init).

## Iteration — `next()` / `isFinished()` / `reset()` / `seek(long)` / `size()`

- `next()` — increments `index`; if still in range, seeks `back` to `permutation[index]`. At the end it simply lets `index` run off, so `isFinished()` becomes true.
- `isFinished()` — `index >= permutation.length`.
- `reset()` — `index = 0`, clears `fresh`, seeks `back` to `permutation[0]`. **Note:** reset does *not* reshuffle; the same permutation is replayed. Re-randomizing is the job of `shuffle()`.
- `seek(long position)` — bounds-checks against `permutation.length`, sets `index`, seeks `back` to `permutation[index]`. Lets external code random-access into the shuffled order.
- `size()` — `permutation.length`.

## Read pass-throughs — `getWeight` / `giveInputs` / `giveOutputs` / `inputCount` / `outputCount`

All delegate straight to `back` — the current backing position (set by the last `seek`) already reflects the permutation, so no remapping is needed at read time.

## Reshuffle — `shuffle()`

The public no-arg `shuffle()` is meant to be called between epochs:

- Resets `index = 0`.
- If `back.size()` no longer matches `permutation.length` (the backing set changed size), rebuilds from scratch via `initPermutation()`.
- Otherwise reshuffles the existing `permutation`, **except** it skips the shuffle when `fresh` is set and `index==0` (i.e. it was just freshly shuffled and untouched) — avoiding a wasted shuffle right after construction.
- Sets `fresh = true`, seeks `back` to the new first index, and returns `this` (fluent).

## Gotchas / dead code

- The `fresh`-skip logic in `shuffle()` is subtle: since `shuffle()` itself sets `index=0` immediately before the check, the `index>0` part of the `!fresh || index>0` condition is effectively always false inside `shuffle()`. The guard therefore reduces to "skip the reshuffle iff already `fresh`" — so the very first `shuffle()` call after construction is a no-op (construction already shuffled). Call it deliberately at epoch boundaries, not expecting it to "shuffle on demand" the first time.
- `clone()` clones the permutation, index, backing cursor, and RNG, giving a fully independent shuffled view.
- `reset()` replays the *same* order — if you want a new order per epoch you must call `shuffle()`, not `reset()`.

## Cross-references

- [`TrainingBatchCursor`](TrainingBatchCursor.md) — the `RandomAccess` contract this implements; see also `RamBuffer.shuffle` for the data-moving alternative.
- `org.siquod.ml.data.DataManagement` — `cloneRandom(Random)` used to isolate the RNG (not in this doc set).
