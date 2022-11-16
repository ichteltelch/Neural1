package org.siquod.ml.metrics;

import java.io.PrintStream;
import java.util.function.Consumer;

public class TrackBest implements DerivedMetricObserver {
	public final Metric metric;
	public static final Consumer<? super TrackBest> PRINT = TrackBest::print;
	Runnable onNewBest;

	public TrackBest(Metric m, Runnable onNewBest) {
		this.metric = m;
		this.onNewBest=onNewBest;
		bestValue = m.biggerIsBetter ? Double.NEGATIVE_INFINITY: Double.POSITIVE_INFINITY;
	}
	public TrackBest(Metric m, Runnable onNewBest, DerivableMetricTracker... sources) {
		this(m, onNewBest);
		for(DerivableMetricTracker t: sources)
			t.addDerived(this);
	}
	public TrackBest(Metric m, Runnable onNewBest, Iterable<? extends DerivableMetricTracker> sources) {
		this(m, onNewBest);
		for(DerivableMetricTracker t: sources)
			t.addDerived(this);
	}
	
	public TrackBest(Metric m, DerivableMetricTracker... sources) {
		this(m, (Runnable)null, sources);
	}
	public TrackBest(Metric m, Consumer<? super TrackBest> onNewBest, DerivableMetricTracker... sources) {
		this(m, sources);
		this.onNewBest=()->onNewBest.accept(this);
	}
	public TrackBest(Metric m, Iterable<? extends DerivableMetricTracker> sources) {
		this(m, (Runnable)null, sources);
	}
	public TrackBest(Metric m, Consumer<? super TrackBest> onNewBest, Iterable<? extends DerivableMetricTracker> sources) {
		this(m, sources);
		this.onNewBest=()->onNewBest.accept(this);
	}

	
	DerivableMetricTracker bestSource;
	double bestValue;
	int bestIteration;
	
	public int bestIteration() {
		return bestIteration;
	}
	public DerivableMetricTracker bestSource() {
		return bestSource;
	}
	public double bestValue() {
		return bestValue;
	}
	@Override
	public synchronized void observe(DerivableMetricTracker source, int iteration, double value) {
		boolean newBest;
		if(metric.biggerIsBetter) {
			newBest = value>bestValue;
		}else {
			newBest = value<bestValue;
		}
		if(newBest) {
			bestValue=value;
			bestIteration = iteration;
			bestSource = source;
			if(onNewBest!=null)
				onNewBest.run();
		}
	}
	public void print() {
		print(System.out);
	}
	public void print(PrintStream out) {
		System.out.println(bestSource.id+" "+bestSource.metric+" at iteration "+bestIteration+": "+bestValue);
	}
	
}
