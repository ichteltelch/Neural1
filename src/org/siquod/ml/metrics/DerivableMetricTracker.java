package org.siquod.ml.metrics;

import java.io.PrintStream;
import java.util.WeakHashMap;

public class DerivableMetricTracker implements MetricObserver{
	
	public WeakHashMap<DerivedMetricObserver, ?> derived = new WeakHashMap<>();
	public final Metric metric;
	public final String id;
	public DerivableMetricTracker(String id, Metric m) {
		this.id = id;
		this.metric = m;
	}

	@Override
	public final void observe(int iteration, double value) {
		doObserve(iteration, value);
		for(DerivedMetricObserver t: derived.keySet())
			t.observe(this, iteration, value);
	}

	public void addDerived(DerivedMetricObserver o) {
		derived.put(o, null);
	}
	
	protected void doObserve(int iteration, double value) {}
	
	
	public MA_MetricTracker movingAverage(int past, int future) {
		return new MA_MetricTracker(id, metric, past, future, this);
	}
	public MetricTracker movingMedian(int past, int future) {
		return new MM_MetricTracker(id, metric, past, future, this);
	}
	public MetricTracker movingRobustAverage(int past, int future, double fraction) {
		return new MRA_MetricTracker(id, metric, past, future, fraction, this);
	}
	public MA_MetricTracker movingAverage(int pastAndFuture) {
		return new MA_MetricTracker(id, metric, pastAndFuture, pastAndFuture, this);
	}
	public MetricTracker movingMedian(int pastAndFuture) {
		return new MM_MetricTracker(id, metric, pastAndFuture, pastAndFuture, this);
	}
	public MetricTracker movingRobustAverage(int pastAndFuture, double fraction) {
		return new MRA_MetricTracker(id, metric, pastAndFuture, pastAndFuture, fraction, this);
	}

	public DerivedMetricObserver printer() {
		return DerivedMetricObserver.printer(this);
	}
	public DerivedMetricObserver printer(PrintStream out) {
		return DerivedMetricObserver.printer(this, out);
	}



}
