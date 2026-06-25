# `DerivedMetricObserver`

Observer interface for downstream consumers that need to know *which source tracker* produced each `(iteration, value)` sample.

Source folder: `Neural1` (package `org.siquod.ml.metrics`).

Up: [index](../_index.md).

## What it is for

`DerivedMetricObserver` is the consumer end of a `DerivableMetricTracker`'s fan-out. When a tracker is fed a value, it notifies every registered `DerivedMetricObserver`, passing not just `(iteration, value)` but also the `DerivableMetricTracker source` that emitted it. That extra `source` argument is what lets a single observer be subscribed to several trackers at once and still know which one fired — exactly what `TrackBest` needs to remember *which* model/source achieved the best value, and what a printer needs to label its output.

Implementors in this package: `TrackBest` (best-so-far selection), `MetricTracker` (a derived tracker that re-emits a smoothed value), and the anonymous printer built by the static factory below.

## The observe contract — `observe(DerivableMetricTracker source, int iteration, double value)`

The single abstract method. Compared to `MetricObserver.observe(int, double)`, it adds the `source` tracker so the consumer can attribute the sample. A `DerivableMetricTracker` calls this on each derived observer from inside its own `observe(...)`.

## Printer factory — `printer(source)` / `printer(source, out)`

Two static helpers build a ready-made observer that logs each sample. They construct an anonymous `DerivedMetricObserver` whose instance initializer immediately calls `source.addDerived(this)` (so it self-subscribes to the given tracker) and whose `observe` prints `"<id> <metric>, iteration <i>: <value>"`. `DerivableMetricTracker.printer()` delegates here.

## Gotchas / dead code

- **Mismatched stream:** `printer(source, out)` accepts a `PrintStream out` parameter but the body ignores it and always writes to `System.err` (`System.err.println(...)`). Passing `System.out` does not redirect the output. See INCONGRUENCIES.
- The printer self-registers in its instance initializer block — a slightly unusual construction side effect; creating one already wires it to the source tracker, so you must keep a reference or it can be garbage-collected (subscriptions are held in a `WeakHashMap`, see `DerivableMetricTracker`).

## Cross-references

- [`DerivableMetricTracker`](DerivableMetricTracker.md) — fans out to these observers; provides `addDerived` and `printer()`.
- [`TrackBest`](TrackBest.md) — the main implementor; uses `source` to record `bestSource`.
- [`MetricObserver`](MetricObserver.md) — the simpler source-side observer without the `source` argument.
