# `TrackBest`

A `DerivedMetricObserver` that remembers the best-so-far metric value, the iteration it occurred at, and which source produced it — the early-stopping / best-checkpoint hook.

Source folder: `Neural1` (package `org.siquod.ml.metrics`).

Up: [index](../_index.md).

## What it is for

`TrackBest` subscribes to one or more `DerivableMetricTracker`s and, for a given `Metric`, keeps the single best value seen across all of them. On every new best it can fire a callback (`onNewBest`) — this is where a training loop saves the current model weights, records the best epoch for early stopping, or prints progress. It records three things about the best: `bestValue`, `bestIteration`, and `bestSource` (the tracker, hence the model/run, that achieved it). Because it watches multiple sources at once, it can pick the best across several candidate models/runs.

Consumers import `TrackBest` to drive best-checkpoint selection while training classifiers.

## What "best" means — `metric.biggerIsBetter`

The constructor seeds `bestValue` to `-∞` if `metric.biggerIsBetter` else `+∞`, so the very first observed value always wins. Every later sample is compared with `>` (bigger-is-better) or `<` (smaller-is-better) accordingly. So the metric's direction flag is the *only* thing that decides whether high or low values are kept — pass a `Metric` with the wrong direction and `TrackBest` will faithfully track the worst value.

## Construction / wiring — the constructor family

Many overloads, all funneling through `TrackBest(Metric m, Runnable onNewBest)` which sets the metric, the callback, and the seed `bestValue`. The rest add source wiring and callback conveniences:

- `(m, onNewBest, DerivableMetricTracker... sources)` and `(m, onNewBest, Iterable<...> sources)` — also call `addDerived(this)` on each source, subscribing this tracker-of-best to all of them.
- `(m, DerivableMetricTracker... sources)` / `(m, Iterable sources)` — no callback (`Runnable` null), just wiring.
- `(m, Consumer<? super TrackBest> onNewBest, sources...)` — accepts a `Consumer<TrackBest>` (gets `this`) instead of a bare `Runnable`; wrapped as `()->onNewBest.accept(this)`. The static `PRINT = TrackBest::print` is the canonical consumer, so `new TrackBest(m, TrackBest.PRINT, sources...)` prints on each new best.

Note the wiring happens in the constructor, so a `TrackBest` is live the moment it is built — but it is held only weakly by each source's `WeakHashMap` (see Gotchas).

## The core — `observe(source, iteration, value)` (synchronized)

For each incoming sample: compute `newBest` via the direction comparison; if it is a new best, update `bestValue`/`bestIteration`/`bestSource` and run `onNewBest` (if non-null). It is `synchronized`, so concurrent sources/threads can feed it safely.

## Accessors and printing — `bestValue()`, `bestIteration()`, `bestSource()`, `print()`

Trivial getters expose the recorded best. `print()` / `print(PrintStream out)` log `"<bestSource.id> <metric> at iteration <bestIteration>: <bestValue>"`.

## Gotchas / dead code

- **`print(out)` ignores `out`:** it builds the line but writes with `System.out.println(...)` rather than `out.println(...)`. Passing a custom `PrintStream` has no effect. See INCONGRUENCIES.
- **Weak subscription:** sources hold `TrackBest` in a `WeakHashMap`; if you don't keep a strong reference, it can be GC'd and silently stop tracking.
- `print()` dereferences `bestSource`, which is `null` until the first observation — calling `print()` before any value is observed throws `NullPointerException`.
- Direction correctness depends entirely on the `Metric` passed in; `TrackBest` does no sanity check that the metric direction matches the sources.

## Cross-references

- [`DerivableMetricTracker`](DerivableMetricTracker.md) — the sources `TrackBest` subscribes to via `addDerived`.
- [`DerivedMetricObserver`](DerivedMetricObserver.md) — the interface implemented; supplies the `source` argument used for `bestSource`.
- [`Metric`](Metric.md) — its `biggerIsBetter` flag defines "best".
- [`MetricTracker`](MetricTracker.md) — smoothing trackers are common `TrackBest` sources (track best of a moving average rather than the noisy raw metric).
