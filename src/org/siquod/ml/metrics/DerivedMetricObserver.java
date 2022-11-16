package org.siquod.ml.metrics;

import java.io.PrintStream;

public interface DerivedMetricObserver {
	public static DerivedMetricObserver printer(DerivableMetricTracker source) {
		return printer(source, System.out);
	}
	public static DerivedMetricObserver printer(DerivableMetricTracker source, PrintStream out) {
		return new DerivedMetricObserver() {
			{
				source.addDerived(this);
			}
			@Override
			public void observe(DerivableMetricTracker source, int iteration, double value) {
				System.err.println(source.id+" "+source.metric+", iteration "+iteration+": "+value);
				
			}
		};
	}
	public void observe(DerivableMetricTracker source, int iteration, double value);
}
