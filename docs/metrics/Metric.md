# `Metric`

A named, directional training quantity (loss, accuracy) that knows whether bigger or smaller is better.

Source folder: `Neural1` (package `org.siquod.ml.metrics`).

Up: [index](../_index.md).

## What it is for

`Metric` is the value object that identifies *what* is being measured during training — accuracy, loss, train-accuracy, train-loss, or a derived variant of one of these. It carries two pieces of state: a human-readable `name` and a `boolean biggerIsBetter`. The `biggerIsBetter` flag is the only "logic" in the class: it tells consumers (notably `TrackBest`) which direction counts as an improvement, so the same comparison code works for both accuracy-style metrics (maximize) and loss-style metrics (minimize). A `Metric` is immutable (both fields are `final`) and is shared as a constant; the actual measured numbers live in the trackers, not here.

Four canonical instances are predefined as `public static final`:

- `ACCURACY` (`"Accuracy"`, bigger is better)
- `LOSS` (`"Loss"`, smaller is better)
- `TRAIN_ACCURACY` (`"Train accuracy"`, bigger is better)
- `TRAIN_LOSS` (`"Train loss"`, smaller is better)

Consumers import `Metric` directly to label and configure the metrics they track while training classifiers.

## Construction and identity — `Metric(name, biggerIsBetter)`, `toString()`

The constructor just stores `name` and `biggerIsBetter`. `toString()` returns `name`, so a metric prints as its label in tracker/`TrackBest` output. Note there is no `equals`/`hashCode` override — identity is reference identity, so two separately constructed metrics with the same name are *not* equal (see Gotchas).

## Derived-metric factories — `movingAverage`, `movingMedian`, `movingRobustAverage`

These three methods do **not** compute anything. Each returns a *new* `Metric` whose `name` is a decorated version of the original (e.g. `"MA(2,2) of Loss"`, `"MM(2,1) of Accuracy"`, `"MRA(2,2,0.5) of Loss"`) and which inherits the same `biggerIsBetter` direction. They exist purely so that a smoothed/derived tracker can label its output metric consistently. The arguments mirror the corresponding tracker factories on `DerivableMetricTracker`:

- `movingAverage(past, future)` → label `MA(past,future)`
- `movingMedian(past, future)` → label `MM(past,future)`
- `movingRobustAverage(past, future, fraction)` → label `MRA(past,future,fraction)`

In practice these are invoked by the tracker subclasses (e.g. `MA_MetricTracker` passes `m.movingAverage(past, future)` up to its `MetricTracker` super-constructor), not by user code directly.

## Gotchas / dead code

- No `equals`/`hashCode`: metrics are compared by reference. The four constants are the intended canonical identities; constructing a fresh `new Metric("Loss", false)` yields a distinct object.
- The `movingAverage`/`movingMedian`/`movingRobustAverage` methods here are *name builders only* — they smooth nothing. The real smoothing lives in the matching `DerivableMetricTracker` factories and their `*_MetricTracker` subclasses. The two sets of methods share names but live on different classes; don't confuse them.
- `biggerIsBetter` is the contract relied on by `TrackBest`; a metric constructed with the wrong direction will silently make `TrackBest` pick the worst value as "best".

## Cross-references

- [`DerivableMetricTracker`](DerivableMetricTracker.md) — holds a `Metric` and accumulates its measured values; its `movingAverage`/`movingMedian`/`movingRobustAverage` factories produce derived trackers whose metrics come from the namesake methods here.
- [`TrackBest`](TrackBest.md) — reads `metric.biggerIsBetter` to decide what "best" means.
- [`MetricTracker`](MetricTracker.md) — derived trackers re-label their output `Metric` via the factories above.
