package org.siquod.ml.metrics;

@FunctionalInterface
public interface MetricObserver {
	public void observe(int iteration, double value);
}
