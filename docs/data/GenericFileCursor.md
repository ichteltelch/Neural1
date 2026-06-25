# `GenericFileCursor`

A `RandomAccess` cursor that streams training examples from a flat binary file of big-endian 32-bit floats, with several output-encoding modes (fixed vector, one-hot, scalar label, or in-file outputs) and optional in-file or normalized weights.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

`GenericFileCursor` is a file-backed *source* in the cursor pipeline. The file is a tightly packed sequence of fixed-size records; each record is `inputFloatsPerSet` input floats, optionally `outputFloatsInFile` output floats, optionally one weight float — every value a 4-byte big-endian IEEE-754 float. It implements `TrainingBatchCursor.RandomAccess` and `AutoCloseable`.

Because records are fixed width, `size()` (`entryCount`) is computed directly from the file length, and `seek` can jump by skipping bytes. The cursor keeps a single "current record" decoded into `inputValues` (and `outputValues` when applicable); the read methods copy from those buffers. Typical use: construct over a file, optionally `ramBuffer()`/`ramBufferAndClose()` to pull the whole thing into memory, then drive it like any other cursor.

The class supports several **output modes**, selected by which constructor is used:

- **In-file outputs** — outputs are stored in the file alongside inputs (`outputFloatsInFile > 0`).
- **Fixed output vector** — every example gets the same caller-supplied `fixedOutput[]`.
- **One-hot** — a class index drives a one-hot vector of width `oneHot`.
- **Scalar label** — a single output equal to the class index (`oneHot==0`).

## Constructors — output-mode selection

- `GenericFileCursor(File, int output, int oneHot, int floatsPerSet, Boolean normalizeWeights)` — label/one-hot mode. `output` is the class index; `oneHot` is the one-hot width (or `0` for a scalar-label output). No outputs are read from the file (`outputFloatsInFile=0`, `outputValues=null`).
- `GenericFileCursor(File, int inputFloatsPerSet, int outputFloatsPerSet, Boolean normalizeWeights)` — in-file outputs. Sets `output=-1`, `oneHot=-1`, allocates `outputValues`, and reads `outputFloatsPerSet` output floats per record.
- `GenericFileCursor(File, double[] fixedOutput, int floatsPerSet, Boolean normalizeWeights)` — fixed-output mode. Every example returns the same `fixedOutput` vector; nothing output-related is read from the file.

All three set `weightFloatsPerSet = (normalizeWeights==null ? 1 : 0)` — i.e. a per-record weight float is present in the file **iff** `normalizeWeights` is `null` — and call `reset()` to open the file and load the first record.

## Weight semantics — `getWeight()` and `normalizeWeights`

The `Boolean normalizeWeights` (note: a nullable `Boolean`) tri-states the weight handling:

- `null` — weights are read from the file (one float per record, after the outputs); `getWeight()` returns that `weight`.
- `true` — no weight in file; `getWeight()` returns `1/entryCount` (each example normalized so the whole file sums to ~1).
- `false` — no weight in file; `getWeight()` returns `1`.

## Opening & lifecycle — `reset()` / `close()` / `clone()`

- `reset()` — closes any open stream, opens a fresh `BufferedInputStream(FileInputStream)`, recomputes `entryCount = fileLength / (recordFloats * 4)` where `recordFloats = inputFloatsPerSet + outputFloatsInFile + weightFloatsPerSet`, sets `position=0`, and loads the first record (if any). `IOException`s are caught and printed, not propagated.
- `close()` — closes the underlying stream (`AutoCloseable`).
- `clone()` — constructs a new cursor over the same file via the **label/one-hot constructor**. See Gotchas: this loses fixed-output / in-file-output configuration.

## Iteration & decoding — `next()` / `isFinished()` / `loadNextDataPoint()` / `readFloat`/`readInt`

- `next()` — increments `position`; if not finished, decodes the next record.
- `isFinished()` — `position >= entryCount`.
- `loadNextDataPoint()` — reads `inputValues.length` input floats, then `outputFloatsInFile` output floats, then (if `weightFloatsPerSet==1`) the weight float, all sequentially from the stream.
- `readFloat`/`readInt` (static) — decode one big-endian 32-bit value from four `InputStream.read()` calls (`<<24 | <<16 | <<8 | …`), then `Float.intBitsToFloat`.

## Reads — `giveInputs` / `giveOutputs` / `inputCount` / `outputCount`

- `giveInputs` — copies the decoded `inputValues` into the caller's array.
- `giveOutputs` — branches on mode: copy `outputValues` (in-file); else copy `fixedOutput`; else if `oneHot==0` write the scalar `output` into slot 0; else write a one-hot (`Arrays.fill(0..oneHot, 0)` then `outputs[output]=1`).
- `inputCount()` = `inputFloatsPerSet`; `outputCount()` = `outputFloatsPerSet`.

## Random access — `size()` / `seek(long)`

- `size()` — `entryCount`.
- `seek(long position)` — if seeking backward, `reset()`s to the start first. If seeking forward, computes the byte distance to skip (`(delta-1) * 4 * recordFloats`), drains it from the buffered stream (falling back to a single `read()` when `skip` returns 0, to make progress at EOF/short reads), then decodes the target record and updates `position`. Seeking to the current position is a no-op.

## RAM buffering — `ramBufferAndClose()`

Calls the inherited `ramBuffer()` to materialize the whole file into a `RamBuffer`, then `close()`s the file, returning the in-memory cursor. The convenient one-shot for "load this file into RAM and stop touching disk".

## Gotchas / dead code

- **`clone()` is mode-lossy.** It always calls the *label/one-hot* constructor (`output, oneHot, inputFloatsPerSet, normalizeWeights`). A cursor created in fixed-output or in-file-output mode will clone into a *different* output mode (likely wrong outputs / dimension mismatch). Suspected bug — see Incongruencies.
- **`outputFloatsPerSet` in the label constructor** is computed as `fixedOutput!=null ? fixedOutput.length : oneHot==0 ? 1 : oneHot`, but `fixedOutput` is always null at that point in that constructor (it's only set by the fixed-output constructor), so the first branch is dead there.
- **Errors are swallowed.** `reset`, `loadNextDataPoint`, and `seek` catch `IOException` and merely `printStackTrace()`; a read failure leaves the cursor in a partially-decoded state rather than failing loudly.
- **Endianness / partial reads.** `readInt` doesn't guard against `read()` returning `-1` (EOF), which would corrupt the decoded value rather than signal end-of-stream; correctness relies on `entryCount` being exact.
- Commented-out dead code: an old `giveOutputs0Or1(int)` method (with a self-doubting `//WHAT???`) and an inline NaN-check `println` in `readFloat`. The `position += 1` comment "(red,green,blue)" is a leftover from an image-pixel origin.
- The `seek` byte-skip math reloads via `loadNextDataPoint()` after positioning, so the post-seek current record is the one *at* `position`.

## Cross-references

- [`TrainingBatchCursor`](TrainingBatchCursor.md) — the `RandomAccess` contract; `ramBuffer()` (used by `ramBufferAndClose`), `singleton`, `empty`.
- [`ShuffledCursor`](ShuffledCursor.md) — commonly wraps a file/RAM source to shuffle per epoch.
- `org.siquod.ml.data.TrainingBatchCursor.RamBuffer` — the in-memory target of `ramBufferAndClose()`.
