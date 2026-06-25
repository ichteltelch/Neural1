# `QuadraticInteraction`

A layer that augments its input with learned pairwise (second-order) feature products, then maps the result onward.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

Lets the net model multiplicative feature interactions cheaply. It projects the input into two low-rank factor sets (`left`, `right`) via sub-modules, multiplies them to form product features, concatenates those products onto the (copied) original input, and runs an "after" sub-module on the concatenation to produce the output. The number and shapes of products are defined by `Kernel`s.

## Construction — use the builder

Don't call the package-private constructor directly; build via:

```
QuadraticInteraction.b()        // -> QuadraticInteractionBuilder
    .kernel(left, mid, right)   // products of an L×mid by mid×R factorization
    .symmetricKernel(outer, inner)
    .repetitions(n).scale(s)
    .leftModule(f).rightModule(f).outputModule(f)
    .build();
```

See [QuadraticInteractionBuilder](QuadraticInteractionBuilder.md) for the config surface. `QuadraticInteraction.b()` is the entry point.

## Kernels — `QuadraticInteraction.Kernel`

A `Kernel` specifies one block of interactions: `leftDim × midDim × rightDim` (or a `symmetric` `outerDim × innerDim` block whose products are the upper triangle, length `outer*(outer+1)/2`), a `repetitions` count, the start offsets into the left/right/output vectors, and a `scaleDown` factor applied to each product (keeps products small). The builder lays kernels out end-to-end and tracks the total left/right/output lengths.

## Role in a net

`allocate` derives the left/right/products interfaces from the input's `TensorFormat` (input and output must match except in the channel dimension). The `left`, `right`, and `after` sub-modules come from the factories (default `Dense`). Forward: run left+right modules → copy input into products + append scaled interaction sums → run after module. Sub-modules are exposed via `getSubmodules()`.

## Gotchas / dead code

- `dontBackprop(String phase)` freezes the interaction gradients for a phase.
- The product features are appended **after** the verbatim input channels, so the after-module sees `inputChannels + outLength` features.

## Cross-references

- [QuadraticInteractionBuilder](QuadraticInteractionBuilder.md) — the configuration API (kernels, factories, scale).
- [Dense](Dense.md) — default left/right/after factor module.
- [StackModule](StackModule.md) — where the layer is placed.
