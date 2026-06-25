# `InOutModule`

Base interface for any `Module` that has one named input interface and one named output interface — the building block from which feed-forward layers are made.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules`).

Up: [index](../_index.md).

## What it is for

Every layer you stack into a net (`Dense`, `Nonlin`, `Dropout`, `BatchNorm`, ...) implements `InOutModule`. It extends the lower-level `Module` interface and adds the contract that the module exposes exactly one input (`getIn()`) and one output (`getOut()`) `Interface`. A `StackModule` chains these together by wiring each module's output to the next module's input.

## Interface wiring — `allocate(ia, String in, String out)`

The default `allocate(InterfaceAllocator ia, String in, String out)` is how a container tells the module which named interfaces to use as its input and output: it pushes a `{"in"->in, "out"->out}` name map onto the allocator, calls the module's own `allocate(ia)` (which then does `ia.get("in")` / `ia.get("out")`), and pops. Callers normally use this rather than `allocate(ia)` directly.

## Config surface — `getIn()`, `getOut()`, `dt()`, `shift()`, `copy()`

- `getIn()` / `getOut()` — the resolved input/output `Interface` (size and `TensorFormat`). Valid after allocation.
- `dt()` — time offset of the input the module reads (for recurrent/sequence nets; `0` for plain layers).
- `shift()` — spatial shift, used when the module sits inside a convolution; `null` otherwise.
- `shift(int...)` / `dt(int)` — default to throwing `UnsupportedOperationException`; only modules that support convolution/time shift (e.g. `Dense`) override them.
- `copy()` — returns an `InOutModule`; required so a `StackModule` can be deep-copied.

## Gotchas / dead code

`shift(...)` and `dt(...)` defaults throw, so calling them on a module that does not override them is a runtime error, not a compile error.

## Cross-references

- [StackModule](StackModule.md) — wires `InOutModule`s in sequence.
- [Dense](Dense.md), [Nonlin](Nonlin.md), [Dropout](Dropout.md), [BatchNorm](BatchNorm.md), [BatchReNorm](BatchReNorm.md), [QuadraticInteraction](QuadraticInteraction.md) — implementations.
- `InOutBiasModule`, `InOutCastLayer`, `InOutCastFactory`, `InOutFactory` — related sub-interfaces/factories (not documented here).
