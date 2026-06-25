# `StackModule`

A sequential container of layers, each with an input and output — the backbone a `FeedForward` net is built from.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

`StackModule` is itself an `InOutModule`. You build a net by repeatedly calling `addLayer(...)` to append layers; each layer's output becomes the next layer's input, with named hidden interfaces (`"hidden Layer #N"`) allocated between them and `"in"`/`"out"` at the ends. It also supports residual-style **shortcuts** that copy activations from one hidden interface to a later one.

## Adding layers — `addLayer(...)`, `addFinalLayer(...)`

- `addLayer(int outSize, InOutModule... ms)` / `addLayer(TensorFormat outSize, InOutModule... ms)` — append one or more modules that produce a hidden interface of the given size. With several modules, all share the same output size (a sub-stack).
- `addLayer(InOutModule m, Interface out)` / `addLayer(InOutModule m, TensorFormat outSize)` — append a single module with an explicit output interface (named `"hidden Layer #N"`).
- `addLayer(InOutCastFactory... fs)` — append layers produced from the previous hidden interface (used for `Dropout` etc., which need to see the incoming `Interface`). Cannot be the first layer.
- `addFinalLayer(InOutModule m)` / `addFinalLayer(int|TensorFormat outSize, InOutModule... ms)` — append the last layer; its output is the stack's `"out"`. After this, no more layers may be added (else `IllegalStateException`).
- If you never call `addFinalLayer`, `allocate` auto-appends a `Copy` so the last hidden interface becomes the output.

## Shortcuts (residual connections) — `shortcut(...)`, `endShortcut()`

`shortcut()` / `shortcut(int width)` marks the current hidden interface as a shortcut **source**; `endShortcut()` marks the current one as the **destination**. At allocation a `Copy` module is wired to add the source activations into the destination, provided widths match (`width==-1` means "match counts"). Mismatched shapes are silently skipped.

## Execution order — `makeExec()`, `getSubmodules()`

`makeExec()` flattens layers plus their shortcut copies into the `exec` list, in order. `forward` runs `exec` front-to-back; `backprop`/`gradients` run it back-to-front. `getSubmodules()` returns `exec`.

## Gotchas / dead code

- `getParamBlocks()` builds a `ParamBlocks` named `"Stack"` but then **returns `null`** (line ~224) — likely a bug; callers relying on it get null instead of the assembled blocks. (Oracle: every sibling module returns the `ret` it built.)
- The shortcut shape loop (lines ~165, ~172) iterates with `i` in the condition (`i<...dims.length-1`) while incrementing `j` — almost certainly a copy-paste error; shortcuts with explicit tensor widths are effectively unverified.
- `addLayer` throws `IllegalStateException` once the final layer exists; order matters.

## Cross-references

- [InOutModule](InOutModule.md) — the element type and the interface `StackModule` implements.
- [Dense](Dense.md), [Nonlin](Nonlin.md), [Dropout](Dropout.md), [BatchNorm](BatchNorm.md) — typical contents of a stack.
- `Copy` — used internally for the auto final layer and shortcuts.
