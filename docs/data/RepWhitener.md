# `RepWhitener`

A `Whitener` decorator that applies one shared base whitener independently to each of several repeated channel-blocks within a single, interleaved feature vector.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

Some feature vectors are a concatenation/interleaving of `repetitions` copies of the same small set of `chs` channels (e.g. the same per-channel measurement taken at several time frames or spatial positions). Fitting and storing one full-length whitener would be wasteful and would treat each repetition as an unrelated feature. `RepWhitener` instead holds a single `base` `Whitener` of width `chs = base.dim()` and applies it `repetitions` times, once per block, so all repetitions share the same learned mean/variance.

Because the layout may be interleaved rather than contiguous, a `RepIndexer` functional interface maps `(repIndex, channelIndex) → flatIndex` into the full vector. This decouples the whitening from the physical memory layout.

Constructed via `Whitener.rep(repetitions, repIndexer)` on a base whitener. Reported `dim()` is `chs · repetitions`.

## Transform — `whiten(in, out)` (`double[]` and `float[]`)

For each repetition `r`:
1. Gather that block's channels into `inBuffer[c] = in[indexer.index(r, c)]`.
2. Run `base.whiten(inBuffer, outBuffer)`.
3. Scatter back: `out[indexer.index(r, c)] = outBuffer[c]`.

Separate `double` and `float` scratch buffers (`inBufferD/outBufferD`, `inBufferF/outBufferF`) are pre-allocated so per-call allocation is avoided. Note `outBuffer` is sized `base.dim()`, which equals `chs` for the diagonal/multivariate Gaussian whiteners.

## Copy — `RepWhitener(RepWhitener)` / `copy()`

`copy()` deep-copies: it copies the base via `base.copy()` and clones all four scratch buffers (so a copy can be used concurrently with the original), while sharing the immutable `repetitions`/`indexer`/`chs`.

## Gotchas / dead code

- `whiten` is **not** safe for `in == out` aliasing in general layouts, and reuses shared mutable scratch buffers, so a single `RepWhitener` instance must not be called concurrently from multiple threads — use `copy()` per thread.
- The output buffer width is taken as `base.dim()`; the scatter loop only writes `chs` entries, so this assumes `base.dim() == chs` (true for the Gaussian whiteners). A base whitener that changes dimensionality would be mishandled.

## Cross-references

- [`Whitener`](Whitener.md) — the base interface; `Whitener.rep(...)` is the factory and `RepIndexer` is declared here for that purpose.
