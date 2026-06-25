# `DataManagement`

Stateless helpers for partitioning a dataset into folds/splits (for cross-validation or train/test division), recombining all-but-one fold, and deep-copying a `Random` so a split is reproducible.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

Pure list-manipulation utilities used when preparing training data: carve a `List<E>` into several parts by ratio (optionally shuffled), and assemble a training set from every part except a held-out one. Generic over the element type `E`, so it works on whatever sample representation the caller uses.

## Splitting — `split(data, …, ratios)` / `split(data, …, parts)`

`split(data, shuffle, double... ratios)` divides `data` into `ratios.length` consecutive sublists whose sizes are proportional to the given ratios:
- Normalizes by `renorm = data.size() / Σ ratios`, then computes cumulative end indices by rounding `lastEnd + renorm·ratio[i]`. The final part's end index is forced to `data.size()` so rounding never drops or duplicates the tail element.
- If `shuffle`, it first copies and Fisher–Yates–shuffles the data (using `Math.random()`), so parts are random samples rather than contiguous slices.
- Returns the list of sublists.

Overloads:
- `split(data, boolean shuffle, int parts)` — equal-sized parts (all ratios = 1), shuffle driven by `Math.random()`.
- `split(data, Random rand, double... ratios)` and `split(data, Random rand, int parts)` — identical, but the shuffle uses the supplied `Random` (`rand.nextInt`) for **reproducible** splits; passing `rand == null` skips shuffling (contiguous split).

## Recombining folds — `concatExcept(parts, except)`

Concatenates all sublists in `parts` except index `except` into one `List<E>` — the standard "training set = all folds but the validation fold" operation for k-fold cross-validation. It prints a notice (`"dropped some samples from the training set: <size>"`) for the excluded fold.

## Reproducible RNG — `cloneRandom(rand)`

Deep-copies a `java.util.Random` by serializing it to a byte array and deserializing, yielding an independent generator at the *same* internal state. Useful to replay an identical shuffle/split sequence (e.g. for the same split across runs or to fork a stream). Returns `null` and prints the stack trace on `IOException`/`ClassNotFoundException`.

## Gotchas / dead code

- The `shuffle`/`Math.random()` overloads are non-deterministic; prefer the `Random`-seeded overloads for reproducible experiments.
- `concatExcept` emits a `System.out.println` side effect rather than returning the dropped count — noisy in batch runs and the "dropped some samples" wording is misleading (the fold is intentionally held out, not lost).
- `cloneRandom` swallows exceptions and returns `null`; callers must null-check.
- `main` is a demo splitting the alphabet into 14 shuffled parts.

## Cross-references

- `TrainingBatchCursor` (Neural1) — the streaming counterpart used once data is split; these helpers operate on plain `List`s upstream of cursor construction.
- [`Whitener`](Whitener.md) — fit on the training split produced here, then applied to all splits.
