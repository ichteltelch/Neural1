# `FeedForward`

A feed-forward neural net wrapper: it ties a stack of `Module`s + a loss layer to named input/output/target/loss `Interface`s, then hands out an `Eval` (to run a trained net) and a `NaiveTrainer` (to train one). This is the main entry point a consuming application uses for its classifiers.

Source folder: `Neural1` (package `org.siquod.ml.neural1.net`).

Up: [index](../_index.md).

## What it is for

A `FeedForward` is the top-level container for one model. You give its constructor the network body (`InOutModule net`), a `LossLayer`, and the input/output (and optionally target) tensor shapes; it builds four named `Interface`s — `in`, `out`, `target`, `loss` — and, on `init()`, allocates all activation buffers and parameter slots so it knows `paramCount`, `activationCount`, etc.

It is **stateless with respect to weights**: `FeedForward` itself holds no `ParamSet`. Instead it acts as a factory for two inner helpers that each carry their own params:

- **`Eval`** — wraps a `ParamSet` (the trained weights) and runs inputs through the net (`forward` only) to produce outputs. This is what a consumer calls to *use* a pre-trained classifier.
- **`NaiveTrainer`** — owns its own `ParamSet`, gradient buffer, and an `Updater` (optimizer); runs forward+backprop over batches/epochs to *learn* the weights.

Typical lifecycle: construct `FeedForward` → `init()` (idempotent) → either `getNaiveTrainer(...)` to train, then `trainer.getEvaluator(...)` to get a runnable `Eval`; or, for a net whose weights were loaded from disk, `new FeedForward(...).new Eval(loadedParamSet)`.

## Construction & allocation — constructors, `init()`

Constructors:

- `FeedForward(InOutModule net, LossLayer lossLayer, int inw, int outw)` — input/output given as plain widths (wrapped in 1-D `TensorFormat`s); target shares the output shape.
- `FeedForward(InOutModule net, LossLayer lossLayer, TensorFormat inw, TensorFormat outw)` — same, but with explicit tensor shapes.
- `FeedForward(..., int inw, int outw, int tw)` / `FeedForward(..., TensorFormat inw, TensorFormat outw, TensorFormat tw)` — separate **target** shape `tw` (for losses whose target tensor differs from the output, e.g. an index instead of a one-hot vector).

All four build `in`/`out`/`target` and a fixed 2-element `loss` interface (slot 0 = data loss, slot 1 = regularization loss).

`init()` (public, `synchronized`, idempotent via the `inited` flag): allocates the interface buffers, walks the module tree to allocate activations and batch-statistics buffers, records `evalActivationCount` (the prefix needed for inference), then allocates target/loss and the loss layer's buffers (`activationCount`), and finally allocates parameters via a `ParamAllocator` to get `paramCount`. It also marks the loss layer and `loss` interface as "don't compute in phase `test`". Returns `this`. `Eval` and `NaiveTrainer` instance-initializers call `init()` themselves, so you rarely call it by hand.

`getNaiveTrainer(int bs, Updater u)` — convenience factory: `new NaiveTrainer(bs, u)`.

## `Eval` — running a trained net

`Eval` is the **inference** object a consumer holds onto. It takes the trained weights and exposes batched/single forward passes. It copies the net and loss layer (`net.copy()`, `lossLayer.copy()`) so each `Eval` has independent inference-time state and is safe to use from its own thread.

Construction:

- `new Eval(ParamSet ps)` — batch size 1.
- `new Eval(ParamSet ps, int maxBatchSize)` — allocates an `ActivationBatch` of that capacity. Throws `IllegalArgumentException("Param set has wrong size")` if `ps.size() != paramCount`.

Running:

- `eval(float[] input, float[] output)` — single sample. Clears activations, applies the optional input whitener, runs `net.forward(ForwardPhase.TESTING, ...)`, and copies the `out` interface into `output`. The loss layer is **not** run (loss/target code is commented out).
- `eval(float[][] input, float[][] output)` — many samples; internally chunks by `maxBatchSize` and calls the range overload.
- `eval(float[][] input, float[][] output, int start, int end)` — one batch: loads `[start,end)` inputs, one `forward`, copies outputs back.

Whitening:

- `inputWhitener(Whitener w)` — installs an input `Whitener` (returns `this` for chaining). When set, every input is `whiten(...)`-ed into an internal buffer before being fed to `in`.

Evaluation / diagnostics over a dataset (all take a `TrainingBatchCursor`):

- `computeLoss(data, bs)` / `computeLoss(data, bs, includeRegLoss)` / `computeLoss(data, bs, includeRegLoss, afterBatch)` / `computeLoss(data, ActivationBatch ass, includeRegLoss, afterBatch)` — streams the cursor in batches, runs forward **and** the loss layer in `TESTING` phase, sums loss slot 0 (plus slot 1 if `includeRegLoss`), and returns the mean per sample. `afterBatch` is an optional `Consumer<ActivationBatch>` callback.
- `computeConfusion(data, bs)` — runs the net over the cursor and builds a `[outputCount][outputCount]` confusion matrix by argmax of target vs. argmax of output. (See `confusionMatrixToString` below for printing.)

Accessors / cloning:

- `getParams()` → the `ParamSet`; `getModel()` → the enclosing `FeedForward`.
- `showWeights()` → a string dump of all `Dense` submodule params (debug).
- `getAnother(boolean copyParams)` → a fresh `Eval` over the same model, sharing or cloning the params, copying the whitener if present. Useful to get a per-thread evaluator.

`FeedForward.confusionMatrixToString(int[][] mat)` (static) — formats a confusion matrix into a column-aligned text table.

## `NaiveTrainer` — training a net

`NaiveTrainer` owns the mutable training state: its own `ParamSet ps` (the weights being learned), a learn-rate-multiplier set `lrm` (default all 1, pushed into the net via `setLearnRateMultipliers`), a gradient `ParamSet grad`, an `Updater u` (defaults to `Rprop` if the constructor arg is null), and two `ActivationBatch`es (`ass` forward, `ess` errors) plus a per-sample importance array `imp`.

Public knobs (fields): `learnRate` (default `.01`), `globReg` (global regularization weight, default `1`), `batchReNormWeight` (default `0.1`).

Construction / init:

- `NaiveTrainer(int batchSize, Updater updater)` — sized to `batchSize`; null updater ⇒ `Rprop`. (Usually obtained via `FeedForward.getNaiveTrainer`.)
- `initSmallWeights(double d)` / `initSmallWeights(double d, Random r)` — fills params with small uniform random values in ±d/2, then calls each module's `initParams`. Returns `this`.

Training:

- `epoch(TrainingBatchCursor data, int batchSize)` — runs one full pass over `data` (resetting it), assembling batches of `batchSize` (≤0 ⇒ use the trainer's batch capacity), calling `endBatch()` per batch, and returning the mean loss over the epoch. This is the normal training call.
- `addToBatch(float[] input, float[] targ, double importance)` — manually stage one weighted sample into the current batch (alternative to driving everything from a cursor).
- `endBatch()` — the core step: forward (`TRAINING`) of net + loss, accumulate loss, seed the loss-error signal per sample (weighted by `imp`), backprop loss then net, compute gradients, update batch-norm statistics, regularize, then `u.apply(...)` the optimizer to the params; clears the gradient and batch counter. Returns the batch's mean loss. (Empty/zero batch returns `1e100`.)

Accessors:

- `getParams()` → the trained `ParamSet`.
- `getEvaluator(boolean copyParams)` → a new `Eval` over these params (share or clone) — the usual hand-off from "trained" to "runnable".
- `currentBatchSize()` → samples staged so far.
- `clone()` — deep-copies params/grad/batches/updater/imp so a trainer can be forked (e.g. parallel search). Implements `Cloneable`.

## Gotchas / dead code

- `FeedForward` holds **no** weights; weights live in `Eval`/`NaiveTrainer` `ParamSet`s. An `Eval` built from a mismatched-size `ParamSet` throws at construction — make sure the `ParamSet` you load came from the same architecture.
- `Eval.eval(...)` skips the loss layer entirely; the loss/target lines are commented out. Use `computeLoss` if you actually need a loss number.
- `init()` is idempotent and self-called by the inner classes, but the `FeedForward` constructors leave a `//init();` commented out — first allocation happens lazily when you create an `Eval`/`NaiveTrainer` (or call `init()` explicitly).
- Big commented-out blocks remain: a dependency-graph/`evalPlan`/`trainPlan` scheme in `init()`, batch-norm-statistics forward passes in `endBatch()`, and several debug index checks. None are active.
- `NaiveTrainer.clone()` sets `r.grad = ps.clone()` (clones from `ps`, not `grad`) — harmless since `grad` is cleared after each `endBatch`, but worth noting if you fork mid-batch. See INCONGRUENCIES.
- `nop()` is an unused empty private method.
- The trainer's `Updater` default is `Rprop`; the commented `SGD` alternative is dead.

## Cross-references

- [`Module`](Module.md) — the `InOutModule net` is a `Module` tree; `FeedForward` drives its `forward`/`backprop`/`gradients`/`allocate` lifecycle.
- [`TensorFormat`](TensorFormat.md) — input/output/target shapes passed to the constructors.
- [`ParamSet`](ParamSet.md) — the weight vector carried by `Eval`/`NaiveTrainer`; `paramCount` must match.
- `LossLayer` (`modules.loss.LossLayer`), `Updater` / `Rprop` (`optimizers`), `ActivationBatch` / `ActivationSet` / `Interface`, `TrainingBatchCursor` and `Whitener` (`org.siquod.ml.data`) — collaborators referenced above; not documented here.
