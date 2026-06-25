# `QuadraticInteractionBuilder`

Fluent builder that configures and constructs a [`QuadraticInteraction`](QuadraticInteraction.md) layer.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

The configuration surface for a quadratic-interaction layer: you declare which interaction kernels to compute, how strongly to scale the products, and which sub-modules produce the left/right factors and the final output, then call `build()`.

## Kernels — `kernel(...)`, `symmetricKernel(...)`, `repetitions(...)`

- `kernel(int left, int mid, int right)` — add an asymmetric interaction block producing `left × right` products from a shared `mid` inner dimension.
- `symmetricKernel(int outer, int inner)` — add a symmetric block (products of a set with itself; upper triangle, `outer*(outer+1)/2` outputs).
- `repetitions(int rep)` — number of independent copies of subsequently-added kernels (set before adding them).

Each added kernel advances internal `leftAt`/`rightAt`/`outAt` offsets, so kernels pack contiguously.

## Sub-modules and scale

- `leftModule(InOutFactory)` / `rightModule(InOutFactory)` / `bothModules(InOutFactory)` — factories that produce the modules computing the left/right factor vectors (default: `Dense` with L2 reg).
- `outputModule(InOutFactory)` — factory for the "after" module run on `[input ++ products]` (default `Dense` + L2 reg).
- `scale(double s)` — sets `scaleDown` applied to each product (default `0.1`).
- `config(Consumer)` — apply a configurator lambda (e.g. the `PRINT` configurator that dumps kernels).

## Build — `build()`

Validates at least one kernel exists (throws `IllegalArgumentException` otherwise) and constructs the `QuadraticInteraction` with the assembled kernels, factories and lengths. `getKernels()` returns the kernel array.

## Gotchas / dead code

- Defaults: `DEFAULT_REG = L2Reg(1e-5)`, `PRODUCE_DEFAULT = Dense(true)+reg`, `PRODUCE_DEFAULT_NOBIAS = Dense(false)+reg`, `scaleDown = 0.1`, `repetitions = 1`.
- `repetitions(...)` and `scale(...)` affect only kernels added *after* the call; ordering matters.

## Cross-references

- [QuadraticInteraction](QuadraticInteraction.md) — the layer this builds; `QuadraticInteraction.b()` returns this builder.
- [Dense](Dense.md) — default factor/output module.
- `InOutFactory`, `L2Reg` — factory and regularizer types.
