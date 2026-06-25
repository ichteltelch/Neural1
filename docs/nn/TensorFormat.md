# `TensorFormat`

An immutable shape/layout descriptor for a multi-dimensional tensor: it records the dimensions and maps multi-index coordinates to flat offsets into an activation buffer.

Source folder: `Neural1` (package `org.siquod.ml.neural1`).

Up: [index](../_index.md).

## What it is for

`TensorFormat` is how the net describes the shape of an `Interface` (input, output, target, intermediate). It is `final` and effectively immutable: a `rank` and an `int[] dims`. At the API level it does two jobs:

1. **Shape declaration** — you pass `TensorFormat`s to the `FeedForward` constructors to say "input is a vector of width N" (`new TensorFormat(N)`) or "input is HxWxC" (`new TensorFormat(h, w, c)`).
2. **Index ↔ flat-offset arithmetic** — given coordinates it computes the linear position in the activation array (column-major: the **first** dim is the fastest-varying), and offers helpers to read/write an `ActivationSet` at those coordinates with automatic bounds-checking.

It implements `RepIndexer` (from `org.siquod.ml.data.RepWhitener`), so it can also drive whitening over the tensor layout.

By convention the **last** dimension is the "channels" axis (see `channels()`, `channelStride()`).

## Construction & basic shape — `TensorFormat(int... dims)`, `count()`, `channels()`

- `TensorFormat(int... dims)` — builds a format from its dimensions; `rank = dims.length`. (Width-only nets just use `new TensorFormat(width)`.)
- `count()` — total number of scalar elements (product of all dims) — i.e. the buffer size this interface needs.
- `channels()` — size of the last dim.
- `channelStride()` — product of all dims **except** the last (the stride between successive channels).
- `rank`, `dims` — public final fields.

## Index arithmetic — `index(...)`

Maps coordinates to a flat offset; returns `-1` when any coordinate is out of range (callers use that as "absent").

- `index(i0)`, `index(i0,i1)`, `index(i0,i1,i2)` — rank-1/2/3 fast paths. Layout is `i0 + dims[0]*(i1 + dims[1]*i2)` (first index varies fastest).
- `index(int[] i, int c)` — general form where the last coordinate is passed separately as `c`.

## Buffered access — `get/set/add(ActivationSet, Interface, ...)`

Convenience read/write into an `ActivationSet` at a multi-index, with bounds checks built in (out-of-range reads return `0`, out-of-range writes are no-ops):

- `get(a, iface, i0[, i1[, i2]])` — read a scalar.
- `get(a, iface, int[] i, int c)` — read via array index + separate channel.
- `add(a, iface, ..., float val)` — accumulate into a position (rank-1/2/3 and array forms).
- `set(a, iface, int[] i, int c, float val)` — overwrite a position.

## Shape transforms — reshape helpers (all return new `TensorFormat`s)

These are pure; they never mutate `this`:

- `withChannels(int c)` / `withChangedChannels(int c)` — same shape but last dim replaced by `c`.
- `flattenIndexAndNext(int firstIndex)` — merges dim `firstIndex` with the next one (rank−1). Rejects the last index.
- `insertUnitIndex(int atIndex)` — inserts a size-1 dim, raising rank by 1.
- `to2D()` — normalizes to rank 2: repeatedly flattens until rank ≤ 2, inserting a leading unit dim if it had collapsed to rank 1. (`//TODO: optimize`.)
- `equalExceptChannels(TensorFormat tf)` — true if all dims except the last match (and ranks equal).

## Equality — `equals`, `hashCode`

Value equality and hashing are based purely on `dims` (via `Arrays.equals`/`Arrays.hashCode`), so two formats with identical dimensions are interchangeable as map keys.

## Gotchas / dead code

- **Layout is first-index-fastest** (`i0 + dims[0]*...`), not the usual row-major C convention — keep this in mind when interpreting flat offsets.
- Out-of-range coordinates are silently tolerated: `index(...)` returns `-1`, `get(...)` returns `0`, `set/add(...)` do nothing. No exceptions for bad indices on the access path (only the reshape helpers throw `IllegalArgumentException`).
- Two `index`/`get`/`add` overloads taking a raw `int[]` (and a `double`-valued `add`) are commented out — dead.
- `to2D()` carries a `//TODO: optimize`.

## Cross-references

- [`FeedForward`](FeedForward.md) — consumes `TensorFormat`s as the shapes of its `in`/`out`/`target` interfaces.
- `Interface` / `ActivationSet` (`org.siquod.ml.neural1`) — the buffers that `get/set/add` operate on.
- `RepIndexer` / `RepWhitener` (`org.siquod.ml.data`) — the whitening hook this implements.
