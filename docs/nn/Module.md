# `Module`

The interface every network layer implements: it knows how to allocate its buffers and parameters, run forward/backprop, and enumerate its submodules — letting a net be composed as a tree of modules.

Source folder: `Neural1` (package `org.siquod.ml.neural1`).

Up: [index](../_index.md).

## What it is for

A `Module` is one piece of a network — a dense layer, an activation, a container, etc. A whole net is a **tree** of modules: a parent (e.g. an `InOutModule` passed to `FeedForward`) returns its children from `getSubmodules()`, and most of the framework's recursive operations have default implementations here that simply walk that tree. So to write a layer you implement the few abstract methods (allocation, forward, backprop, copy) and inherit the recursive plumbing.

At the usage level you don't implement `Module` to use the classifiers — you just need to know that the net `FeedForward` drives is a `Module` tree, and that the lifecycle below (allocate → initializeRun → forward → backprop → gradients → regularize → updateStatistics) is what `FeedForward.Eval` and `NaiveTrainer` invoke on it.

## Allocation & parameter layout — `allocate(...)`, `share`, `getParamBlocks`

- `allocate(InterfaceAllocator ia)` — reserve this module's activation buffers.
- `allocate(ParamAllocator ia)` — reserve its parameter slice (its `ParamBlock` within the global `ParamSet`).
- `allocateStatistics(InterfaceAllocator ia)` *(default: recurse)* — reserve batch-statistics buffers (batch-norm etc.).
- `share(ParamBlocks ps)` / `getParamBlocks()` — parameter sharing/weight tying between modules.

## Inference & training lifecycle — `forward`, `backprop`, `gradients`, `regularize`

These are the per-run operations `FeedForward` calls in order:

- `forward(ForwardPhase phase, ParamSet params, ActivationBatch as, int t, int[] inst)` — compute activations for a batch. The `ForwardPhase` selects behavior (`TESTING` for inference, `TRAINING` for learning).
- `backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst)` — propagate error signals back through activations.
- `gradients(... ParamSet gradients ...)` *(default: recurse)* — accumulate parameter gradients into the gradient `ParamSet`.
- `regularize(String phase, ParamSet params, ParamSet gradients, float globReg)` *(default: recurse)* — add regularization to the gradients.
- `dontComputeInPhase(String phase)` — mark this module skippable in a named phase (e.g. the loss layer in `test`).

(The actual forward/backward math lives in each concrete module; this interface only fixes the signatures.)

## Run setup & statistics — `initializeRun`, `updateStatistics`, `setLearnRateMultipliers`, `initParams`

All have tree-recursing defaults:

- `initializeRun(ActivationBatch as, boolean training)` — reset inference-time state before a forward pass.
- `updateStatistics(ActivationSeq stat, ParamSet params, Function<Integer,Float> owt, float[] weight, int tMin)` — fold this batch into running statistics (batch-norm).
- `setLearnRateMultipliers(ParamSet lrm)` — write per-parameter learn-rate multipliers into `lrm`.
- `initParams(ParamSet p)` / `defaultInitParams(ParamSet p)` — initialize this module's parameters (default recurses into children).

## Tree traversal & copy — `getSubmodules`, `deepSubmodules`, `copy`

- `getSubmodules()` *(abstract)* — direct children; the basis for every recursive default above.
- `deepSubmodules()` / `giveDeepSubmodules(List)` — flatten the whole subtree (self first), used e.g. by `Eval.showWeights()` to find all `Dense` layers.
- `copy()` *(abstract)* — deep copy with **no shared inference-time state**. This is what `FeedForward.Eval` uses (`net.copy()`) so each evaluator/thread is independent.

## Gotchas / dead code

- Most framework methods are `default` and just recurse over `getSubmodules()`, so a leaf module that returns no children gets correct (no-op) behavior for free; only `allocate`, `forward`, `backprop`, `dontComputeInPhase`, `getSubmodules`, and `copy` are mandatory.
- `copy()` must drop shared run-time state but should keep structure/params semantics — required for the per-`Eval` net copy to be thread-safe.
- A static `ExecutorService parallelizer` (cached thread pool) and `joinAll(...)` helper live on the interface for modules that parallelize internally; `EIA` is a shared empty `int[]`.
- A few commented-out methods remain (`declareDependencies`, `wouldBackprop`, `initializeBatch`) — part of the abandoned dependency-graph scheme also seen commented out in `FeedForward.init()`.

## Cross-references

- [`FeedForward`](FeedForward.md) — owns the root `InOutModule` (a `Module`) and drives this lifecycle.
- [`ParamSet`](ParamSet.md) — the parameter/gradient vectors threaded through `forward`/`backprop`/`gradients`; each module owns a `ParamBlock` slice.
- [`TensorFormat`](TensorFormat.md) — describes the shapes of the interfaces a module allocates.
- `ForwardPhase`, `ActivationBatch` / `ActivationSeq` / `ActivationSet`, `InterfaceAllocator` / `ParamAllocator`, `ParamBlocks` (`org.siquod.ml.neural1`) — collaborators in the signatures above.
