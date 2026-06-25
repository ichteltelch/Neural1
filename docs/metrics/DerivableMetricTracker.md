# `DerivableMetricTracker`

A named accumulator for one `Metric`: receives `(iteration, value)` samples and fans them out to any number of derived observers (smoothers, printers, best-trackers).

Source folder: `Neural1` (package `org.siquod.ml.metrics`).

Up: [index](../_index.md).

## What it is for

`DerivableMetricTracker` is the central node of the metrics graph. It bundles:

- a `Metric metric` (what is measured, and its better-direction),
- a `String id` (which model/run/source it belongs to — printed and used by `TrackBest.bestSource`),
- a `WeakHashMap<DerivedMetricObserver, ?> derived` registry of downstream observers.

During a training loop you create one tracker per (model, metric) pair, e.g. a `DerivableMetricTracker("A", Metric.LOSS)`. Each time you measure that metric you push the value in via `observe(iteration, value)`. The tracker can then *derive* further trackers from itself (moving average/median/robust-average) and *fan out* every sample to observers such as `TrackBest` or a printer. It is "derivable" precisely because other observers can be attached to its output stream.

Consumers import this class to accumulate per-epoch metric values while training classifiers and to hang `TrackBest` / smoothing off them.

## Receiving and fanning out — `observe(int, double)` (final) + `doObserve`

`observe(iteration, value)` is `final`. It does two things in order:

1. calls the overridable hook `doObserve(iteration, value)` (a no-op in the base class; subclasses can use it), then
2. iterates `derived.keySet()` and calls each observer's `observe(this, iteration, value)`, passing itself as the `source`.

So pushing a value into a tracker immediately propagates it, with source attribution, to every attached `DerivedMetricObserver`. Because `observe` is `final`, subclasses cannot change the fan-out; they customize via `doObserve` (the base) — or, in the derived-tracker subclasses, by overriding the *`DerivedMetricObserver` version* of `observe` and calling the inherited `MetricTracker.observe(int,double)` to re-emit a computed value (this is the smoothing pattern, see `MetricTracker`).

## Subscription registry — `addDerived(observer)` and `derived`

`addDerived(o)` puts `o` into the `derived` `WeakHashMap` (value `null`). The map being *weak-keyed* means a derived observer is held only weakly: if nothing else references it, it can be garbage-collected and silently stops receiving samples. Callers therefore keep their own strong reference to anything they want to keep alive (e.g. a `TrackBest` or a printer).

## Smoothing factories — `movingAverage`, `movingMedian`, `movingRobustAverage`

Six overloads (each with a `(past, future)` form and a symmetric single-arg `(pastAndFuture)` form) construct and return a *new* derived tracker subclass, already subscribed to `this` as its source:

- `movingAverage(...)` → `MA_MetricTracker`
- `movingMedian(...)` → `MM_MetricTracker`
- `movingRobustAverage(..., fraction)` → `MRA_MetricTracker`

Each derived tracker's `Metric` is the relabeled variant produced by `metric.movingAverage(...)` etc. (e.g. `"MA(2,2) of Loss"`), inheriting the same better-direction. Because each derived tracker is itself a `DerivableMetricTracker`, you can chain further observers (or further smoothers, or a `TrackBest`) onto its smoothed output. The `MA_MetricTracker` implementation keeps a ring buffer of `past+future+1` samples, maintains a running `sum`, recomputes the sum from scratch every 1024 samples to fight float drift, and re-emits `sum/count` (tagged with the *center* iteration, `offset+past`) once the window is full.

## Printer helpers — `printer()` / `printer(out)`

Convenience wrappers delegating to `DerivedMetricObserver.printer(this[, out])`; they build and self-subscribe a logging observer. (Note the underlying printer always writes to `System.err` regardless of `out` — see `DerivedMetricObserver`.)

## Gotchas / dead code

- `derived` is a `WeakHashMap`: attached observers must be strongly referenced elsewhere or they vanish. This is the main footgun of the layer.
- `observe` is `final`; the only base-class extension point is the (no-op) `doObserve`. Derived trackers override the *other* `observe(source, iteration, value)` they inherit via `MetricTracker`'s `DerivedMetricObserver` role.
- The class is concrete (not abstract) — you can instantiate a plain `DerivableMetricTracker` as a raw source even though `MetricTracker` extends it as an abstract intermediate.

## Cross-references

- [`MetricTracker`](MetricTracker.md) — abstract subclass that is *also* a `DerivedMetricObserver`; the smoothing trackers (`MA_/MM_/MRA_MetricTracker`) extend it.
- [`Metric`](Metric.md) — supplies the metric identity and the relabeled metrics for derived trackers.
- [`DerivedMetricObserver`](DerivedMetricObserver.md) — the observer type stored in `derived`.
- [`TrackBest`](TrackBest.md) — a typical observer attached via `addDerived`.
- [`MetricObserver`](MetricObserver.md) — the source-side interface this class implements.
