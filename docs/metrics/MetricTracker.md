# `MetricTracker`

Abstract bridge tracker that is simultaneously a metric *source* (`DerivableMetricTracker`) and a *derived consumer* (`DerivedMetricObserver`) — the base for all smoothing trackers.

Source folder: `Neural1` (package `org.siquod.ml.metrics`).

Up: [index](../_index.md).

## What it is for

`MetricTracker` is the node that sits *between* two trackers in the metrics graph. It `extends DerivableMetricTracker` (so it has its own output stream and `derived` registry) **and** `implements DerivedMetricObserver, MetricObserver` (so it can subscribe to an upstream tracker and receive that tracker's samples with source attribution). This dual nature is exactly what a smoother needs: consume raw samples from a source, compute a smoothed value, and re-emit it downstream.

It is abstract — it provides no measurement logic itself. The concrete implementations are `MA_MetricTracker` (moving average), `MM_MetricTracker` (moving median), and `MRA_MetricTracker` (moving robust average), all created via the factory methods on `DerivableMetricTracker`.

## Construction — `MetricTracker(id, m)` and `MetricTracker(id, m, source)`

- `MetricTracker(id, m)` just calls `super(id, m)` (the `DerivableMetricTracker` constructor); a standalone tracker with no upstream.
- `MetricTracker(id, m, DerivableMetricTracker source)` calls `super(id, m)` and then `source.addDerived(this)`, **wiring itself as a derived observer of `source` at construction time**. This is the constructor the smoothing subclasses use, so creating an `MA_MetricTracker(...)` immediately subscribes it to its source's output stream.

## The smoothing pattern (how subclasses use the dual role)

A concrete subclass overrides `observe(DerivableMetricTracker source, int iteration, double value)` (the `DerivedMetricObserver` method) to ingest each upstream sample, updates its internal window/buffer, and when it has a result calls the inherited `observe(int iteration, double value)` (the `DerivableMetricTracker`/`MetricObserver` method, which is `final`) to push the *smoothed* value onward to its own derived observers. Thus one object both receives raw samples and emits computed ones, tagged with the appropriate (often center) iteration. `MA_MetricTracker` is the worked example: ring buffer + running sum, re-emits `sum/count` once the window fills.

## Gotchas / dead code

- The class body is essentially empty apart from the two constructors; almost all behavior is inherited or supplied by subclasses. A large commented-out `main` demonstrating moving-average/median/robust-average wiring and a `TrackBest`/printer hookup is left in the source as documentation-by-example.
- Implementing *both* `observe` overloads (the `(int,double)` source-side one inherited `final` from `DerivableMetricTracker`, and the `(source,int,double)` derived-side one from `DerivedMetricObserver`) on the same object is the crux — easy to conflate. The derived-side one is the *input*; the source-side `final` one is the *output*.
- `implements MetricObserver` is redundant in that `DerivableMetricTracker` already implements `MetricObserver`; it is restated here harmlessly.

## Cross-references

- [`DerivableMetricTracker`](DerivableMetricTracker.md) — superclass; provides the output stream, `addDerived`, and the factory methods that build the concrete subclasses.
- [`DerivedMetricObserver`](DerivedMetricObserver.md) — the consumer interface this class implements to read its upstream source.
- [`Metric`](Metric.md) — derived trackers carry a relabeled metric (`"MA(...) of ..."`).
- `MA_MetricTracker` / `MM_MetricTracker` / `MRA_MetricTracker` — the concrete smoothers (not separately documented).
