# `MetricObserver`

Functional interface for anything that wants to be fed `(iteration, value)` samples of a metric.

Source folder: `Neural1` (package `org.siquod.ml.metrics`).

Up: [index](../_index.md).

## What it is for

`MetricObserver` is the *push* entry point of the metrics layer. It is the sink the training loop calls each time a new measurement is available: at iteration `i`, value `v`, you call `observer.observe(i, v)`. It is a `@FunctionalInterface`, so a lambda or method reference can serve as one, but in practice the concrete implementor is `DerivableMetricTracker` (and therefore `MetricTracker`), which both *receives* samples via this interface and *forwards* them to its derived observers.

This is the "source-side" observer (raw samples by iteration index), as opposed to `DerivedMetricObserver`, whose `observe` also carries the *source tracker* that produced the value.

## The single method — `observe(int iteration, double value)`

One method, no return value. `iteration` is the epoch/step index the value belongs to; `value` is the measured number for the metric. The implementor decides what to do — accumulate it, smooth it, fan it out, or compare it against a best-so-far.

## Gotchas / dead code

- This interface deliberately does **not** name the metric or the source. The metric identity is implied by *which* `DerivableMetricTracker` you call. If you need the source identity in your callback (e.g. for printing or best-tracking), you want `DerivedMetricObserver` instead.
- Being a `@FunctionalInterface` it is trivially lambda-able, but every real implementation in this package routes through `DerivableMetricTracker.observe`, which is `final` and adds the fan-out behavior.

## Cross-references

- [`DerivableMetricTracker`](DerivableMetricTracker.md) — the canonical implementor; its `observe(int, double)` is `final` and forwards to derived observers.
- [`DerivedMetricObserver`](DerivedMetricObserver.md) — the "derived-side" observer that additionally receives the source tracker.
- [`MetricTracker`](MetricTracker.md) — implements both observer interfaces.
