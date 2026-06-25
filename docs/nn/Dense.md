# `Dense`

A fully-connected (linear) layer: output = weight matrix · input (+ optional bias).

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

The standard learnable layer. Maps an input vector of `in.count` elements to an output of `out.count` elements through a learned weight matrix, plus an optional bias per output. The output size is determined by the hidden/output interface you give it when adding it to a `StackModule`. Almost always followed by a `Nonlin` activation.

## Construction — `Dense(...)`

- `Dense()` — bias on.
- `Dense(boolean useBias)` — toggle the bias term.
- `Dense(int dt, int[] shift, boolean... useCoordinateBias)` / `Dense(boolean useBias, int dt, int[] shift, boolean...)` — convolutional / time-shifted variants: `dt` reads the input from a different time step, `shift` makes it a convolution kernel (must then sit inside a convolution), and `useCoordinateBias[i]` adds an extra learned per-output bias scaled by coordinate activation `i`.

## Fluent config

- `biasInit(float)` — initial bias value (default 0).
- `regularizer(Regularizer)` — weight regularizer (e.g. L2); applied in `regularize`.
- `dt(int)` / `shift(int...)` — set time offset / convolution shift after construction.
- `parallel(int threads)` — split forward/backprop/gradient work across threads.
- `dontBackprop(String phase)` — freeze the layer (no input-error propagation) in the named phase.
- `setCoordinateBiases(float[])` — supply the coordinate-activation values for coordinate biases.

## Parameters / init

`allocate(ParamAllocator)` allocates a `weights` block of `in.channels * out.channels`, plus `bias` (if used) and any `coordinateBias` blocks. `initParams` scales weights by `sqrt(2/(in+out))` (He/Glorot-style) and sets bias to `biasInit`. `getBias()` exposes the bias block. `showParams(ParamSet)` pretty-prints the matrix and bias.

## Gotchas / dead code

- Implements `InOutBiasModule`. `getParamBlocks()` names itself `"FullyConnected"` (the class's old name; commented-out `useResidual` code remains).
- `share(...)` sets `biasInit=Float.NaN`, which makes `initParams` skip bias initialization for shared layers (intentional, but subtle).
- Forward skips zero inputs as an optimization, so it assumes additive accumulation into a freshly-zeroed output.

## Cross-references

- [StackModule](StackModule.md) — how `Dense` is placed; output size comes from the layer's interface.
- [Nonlin](Nonlin.md) / [ParameterizedNonlin](ParameterizedNonlin.md) — typically applied after.
- `Regularizer`, `L2Reg` — weight regularization.
- [QuadraticInteractionBuilder](QuadraticInteractionBuilder.md) — uses `Dense` as its default factor/output module.
