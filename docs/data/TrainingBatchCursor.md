# `TrainingBatchCursor`

The central iterator abstraction of the data layer: a forward cursor over training examples that can be reset, cloned, batched, randomly accessed, concatenated, RAM-buffered, shuffled, remapped, whitened, and transformed — all composed from this one interface and its `RandomAccess` sub-interface.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

A `TrainingBatchCursor` extends [`TrainingDataGiver`](TrainingDataGiver.md) (which exposes the *current* example's inputs/outputs/weight and the vector dimensions) and adds **iteration**: `next()` advances, `isFinished()` reports end-of-sequence, and `reset()` restarts. It is the unit that a fitting/training algorithm consumes: loop `while(!isFinished()){ giveInputs(...); giveOutputs(...); getWeight(); next(); }`, then `reset()` for the next epoch.

The interface is deliberately thin (four abstract methods plus those inherited), but it hosts a large library of `default`/`static` methods and nested classes that let cursors **compose**. The typical pipeline is: a source cursor (file- or RAM-backed) → optionally `ramBuffer()` it into memory → wrap in a `ShuffledCursor` for per-epoch shuffling → `transformInputs`/`transformOutputs`/`whitened`/`remap`/`scaleWeights` to massage features and weights → `split`/`subsequence`/`fillToModulus` to carve train/val sets and pad to batch size → feed to the algorithm. Each wrapper is itself a `TrainingBatchCursor`, so they stack.

Two capability tiers exist:

- **Plain `TrainingBatchCursor`** — forward-only iteration; cannot report its length or jump.
- **`TrainingBatchCursor.RandomAccess`** (nested interface) — additionally knows its `size()` and supports `seek(position)`. This unlocks shuffling, splitting, subsequencing, modulus-padding, and indexed concatenation. Most decorators come in both a plain and a `_RA` variant so the random-access capability is preserved through the wrapper.

Key state lives in the implementations, not the interface; the interface mostly defines *how the pieces fit together*.

## Core iteration contract — `next()` / `isFinished()` / `reset()` / `clone()`

- `next()` — advance to the next example, or step off the end. After this the read methods describe the new current example (or are illegal if now finished).
- `isFinished()` — true once the sequence is exhausted. While true, the inherited `giveInputs`/`giveOutputs`/`getWeight` must not be called until `reset()`.
- `reset()` — restart from the beginning. The cursor must re-present the same items (the Javadoc allows a subclass to *reload* with different data between training runs, but not mid-batch).
- `clone()` (interface extends `Cloneable`) — produce an independently iterable copy. Used heavily by `subsequence`/`split`/`concat` so the original cursor's position isn't disturbed by carving out views.

## Random access tier — `RandomAccess`: `size()` / `seek(long)` / `clone()`

`RandomAccess` is a `TrainingBatchCursor` with a known, fixed length:

- `size()` — number of items.
- `seek(long position)` — position the cursor at an arbitrary index quickly. Convention enforced by some implementations: position must be in range; note the recurring (mis-worded) guard `"Seek position must mut be negative"` actually rejects *negative* positions.
- `clone()` — covariantly returns `RandomAccess`; the copy may or may not start in the reset state.

It also supplies the splitting/subsequencing/padding/transform conveniences below.

## Splitting & subranges — `split(int)` / `split(double...)` / `subsequence(long,long)` / `fillToModulus(int,boolean)`

All on `RandomAccess`:

- `split(int parts)` — divides into `parts` contiguous, roughly equal-length cursors via `subsequence`, using integer boundary arithmetic `(total*part)/parts`.
- `split(double... ratios)` — divides into chunks whose lengths are proportional to `ratios` (renormalized to `size()`); rounds interior boundaries and forces the last boundary to exactly `size()` so nothing is lost. Typical use: `split(0.8, 0.2)` for a train/validation partition.
- `subsequence(long start, long end)` — an independent cursor (built on `clone()`) restricted to `[start, end)`. Clamps `start` to `0` and `end` to `size()`; returns the shared `empty(...)` cursor when the range is empty. The returned anonymous `RandomAccess` tracks its own local `at`, translating `seek`/`reset` onto the cloned backing cursor (`orig.seek(fStart+position)`), and its own `subsequence` composes by offsetting into the original — so subranges of subranges don't nest wrappers unboundedly.
- `fillToModulus(int modulus, boolean roundUpNotDown)` — make the length a multiple of `modulus` (e.g. a batch size). It `concat`s the cursor with itself enough times, then `subsequence`-truncates the overflow. Useful so the last batch is full.

## Remapping & weights — `remap(...)` / `scaleWeights(double)` and `RemappedCursor`

- `remap(double weightScale, int[] outputMapping, int... inputMapping)` — reorders/selects input and output indices and scales weights; backed by `RemappedCursor` (and `RemappedCursor.RandomAccess`). A `null` mapping means "pass that side through unchanged"; otherwise the new vector length is the mapping length, and each slot `i` is taken from the original at `mapping[i]` via an internal `buffer`.
- `scaleWeights(double weightScale)` — convenience for `remap` with null mappings; just multiplies every `getWeight()` by `weightScale`.

`RemappedCursor` delegates iteration straight to its `original` and exposes the original as a public field; the `RandomAccess` variant forwards `size`/`seek`.

## Feature/output/weight transforms — `transformInputs/​transformOutputs/​transformWeights(...)`

A family of `default` methods (mirrored on both `TrainingBatchCursor` and `RandomAccess`) that wrap the cursor in a [`TransformedCursor`](TransformedCursor.md) (or `TransformedCursor_RA`). They accept a [`CursorTransformer`](CursorTransformer.md) for inputs and/or outputs and a `DoubleUnaryOperator` for weights, in various combinations. The transformer may change dimension (the wrapper sizes its vectors to the max of before/after). See `TransformedCursor` for the per-example mechanics.

Note: several of the `transformOutputs(...)` overloads actually take *both* an input and an output transformer (the name is misleading — see Gotchas).

## Polynomial feature expansion — `polyInteractionFeatures(...)` and `PolyInteractionCursor_RA`

Static factories that wrap a cursor so its inputs are enriched with polynomial interaction terms up to a given `order` (optionally leaving a `prefixLength`/`suffixLength` of inputs untouched). `order==1` short-circuits and returns the cursor unchanged. The actual expansion lives in `PolyInteractionCursor` (outside this file); `PolyInteractionCursor_RA` is the random-access wrapper that forwards `size`/`seek`/`clone`.

## Concatenation — `concat(...)` and `TrainingBatchCursor_Concat`

- `concat(TrainingBatchCursor...)` / `concat(List)` / `concat(RandomAccess...)` / `concatRandomAccess(List)` — join sequences end-to-end. All parts must agree on `inputCount()` and `outputCount()` (validated in the constructor, which also rejects an empty array). A single-element array is returned as-is. Factories `reset()` the result before returning.
- `TrainingBatchCursor_Concat<B>` — holds the parts and a `currentSequenceIndex`. The key helper `ff()` ("fast-forward") skips over already-finished (including empty) parts, resetting each part as it's entered; `isFinished()` calls `ff()` first so empty parts in the middle are transparent. `clone()` clones every part and re-`concat`s.
- The random-access concat (`TrainingBatchCursor_ConcatRandomAccess`, a local class inside `concat(RandomAccess...)`) precomputes a cumulative `index[]` of part boundaries so `size()` is `O(1)`, `seek` does a binary search (`seqi`) to find the owning part then seeks within it, and `subsequence` walks the parts overlapping `[start,end)`, cloning whole parts and sub-sequencing the partial ones, then re-`concat`s — keeping random access through the join.

## Singletons & empty — `singleton(...)` / `empty(int,int)`

- `singleton(double[] inputs, double[] outputs, double weight)` and the scalar-output overload `singleton(double[] inputs, double output, double weight)` — a one-element `RandomAccess`. A `consumed` flag models position: `next()`/`seek(>0)` mark it finished; `reset()`/`seek(0)` un-consume it. The arrays are *not* defensively copied — the cursor shares them.
- `empty(int inputCount, int outputCount)` — a zero-length `RandomAccess` that is always finished. The read methods (`getWeight`/`giveInputs`/`giveOutputs`/`next`) throw `UnsupportedOperationException("Empty cursor")`; `subsequence` returns itself. Used as the canonical result of degenerate `subsequence` ranges.

## RAM buffering — `ramBuffer()` / `ramBuffer(TrainingBatchCursor)` and `RamBuffer`

- `ramBuffer(of)` — materializes an entire cursor into memory by iterating it once (after `reset()`) and collecting every input array, output array, and weight, then packing them into primitive `double[][]`/`double[]`. If `of` is already a `RamBuffer`, it just `clone()`s it. Hard limit: throws `ArrayIndexOutOfBoundsException` past `Integer.MAX_VALUE` items.
- `RamBuffer` (`RandomAccess`) — the in-memory backing store. Iteration is index arithmetic (`at`), `seek` bounds-checks, `size()` is the array length, and `getWeight()` returns `1` if the weight array is null. `clone()` shares the underlying arrays (cheap, and intentionally so for shuffling).
- `RamBuffer.shuffle(Random)` — Fisher–Yates shuffles the underlying arrays **in place**, which by design also reorders *all clones* (they share the arrays). This is the fast path when you want one canonical buffer reshuffled each epoch; contrast with [`ShuffledCursor`](ShuffledCursor.md), which permutes via an index array over any `RandomAccess` without touching the data.

## Whitening — `whitened(Whitener, Whitener)` and `WhitenedTrainingBatchCursor` / `WhitenedRandomAccess`

Overrides `TrainingDataGiver.whitened(...)` to return an *iterating* whitened wrapper. `WhitenedTrainingBatchCursor<B>` extends `TrainingDataGiver.WhitenedTrainingDataGiver` (which does the actual standardizing of inputs/outputs) and adds `next`/`isFinished`/`reset`/`clone` by delegation. `WhitenedRandomAccess<B>` further adds `size`/`seek`, preserving random access through whitening.

## Aggregate scans — `getTotalWeight()` / `containsNaN()`

Both `reset()` then iterate to the end:

- `getTotalWeight()` — sums `getWeight()` across all items.
- `containsNaN()` — returns true if any weight, input, or output value is `NaN`. A cheap data-sanity check before training. Both leave the cursor finished (they don't reset afterwards).

## Gotchas / dead code

- **Misleading overload names:** `transformOutputs(CursorTransformer inT, CursorTransformer outT)` and `transformOutputs(inT, outT, weightT)` take an *input* transformer as the first argument despite the `transformOutputs` name — copy-paste artifacts of `transformInputs`. Read the parameter list, not the method name.
- **Off-by-name guard:** the `seek` validation message `"Seek position must mut be negative"` (appears in `subsequence`'s anonymous class and in `ConcatRandomAccess`) is double-wrong: it has a typo ("mut") and states the opposite of what it enforces (it rejects negative positions).
- **`RemappedCursor.giveOutputs` uses `ic`, not `oc`:** the output remap loop runs `for(int i=0; i<ic; ++i)` — it iterates over the *input* count when copying outputs. If `inputCount != outputCount` and an `outputMapping` is supplied, this either under-/over-copies or indexes out of range. Looks like a bug (see Incongruencies).
- **`singleton(...)` shares arrays:** no defensive copy; mutating the passed arrays afterward mutates the cursor's data.
- **`RamBuffer.shuffle` mutates shared state:** clones share arrays, so shuffling one shuffles all — surprising unless you know the design intent.
- Commented-out debug `System.out.println()` lines in `RamBuffer.isFinished()` and in `GenericFileCursor`-style readers.
- `usedInputCount`/`usedOutputCount` (inherited) are not overridden anywhere here.

## Cross-references

- [`TrainingDataGiver`](TrainingDataGiver.md) — the read-side super-interface (dimensions, `giveInputs`/`giveOutputs`/`getWeight`, base `whitened`).
- [`ShuffledCursor`](ShuffledCursor.md) — index-permutation shuffler over any `RandomAccess` (alternative to `RamBuffer.shuffle`).
- [`GenericFileCursor`](GenericFileCursor.md) — a `RandomAccess` source that streams examples from a binary file; pairs with `ramBuffer()`/`ramBufferAndClose()`.
- [`TransformedCursor`](TransformedCursor.md) / [`CursorTransformer`](CursorTransformer.md) — the per-example transform layer behind `transformInputs/Outputs/Weights`.
- `org.siquod.ml.data.RemappedCursor`, `PolyInteractionCursor`, `Whitener`, `DataManagement` — referenced/used here but documented elsewhere (or not yet).
