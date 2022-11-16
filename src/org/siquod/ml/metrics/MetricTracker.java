package org.siquod.ml.metrics;



public abstract class MetricTracker extends DerivableMetricTracker implements DerivedMetricObserver, MetricObserver{
	
	public MetricTracker(String id, Metric m) {
		super(id, m);
	}
	public MetricTracker(String id, Metric m, DerivableMetricTracker source) {
		super(id, m);
		source.addDerived(this);
	}
	
	
//	public static void main(String[] args) {
//		Metric m = Metric.ACCURACY;
//		DerivableMetricTracker a = new DerivableMetricTracker("A", m);
//		DerivableMetricTracker b = new DerivableMetricTracker("B", m);
//		MetricTracker ama = a.movingRobustAverage(2, 2, 0.5);
//		MetricTracker bma = b.movingMedian(2, 1);
////		TrackBest best = new TrackBest(m, TrackBest.PRINT, ama, bma);
//		DerivedMetricObserver amaPrint = ama.printer();
////		DerivedMetricObserver bmaPrint = bma.printer();
////		a.observe(0, 1); b.observe(0, 2);
////		a.observe(1, 2); b.observe(1, 1);
////		a.observe(2, 3); b.observe(2, 1);
////		a.observe(3, 4); b.observe(3, 1);
////		a.observe(4, 9); b.observe(4, 1);
////		a.observe(5, 6); b.observe(5, 4);
////		a.observe(6, 7); b.observe(6, 100);
////		a.observe(7, 8); b.observe(7, 3);
////		a.observe(8, 1); b.observe(8, 0);
//		for(int i=0; i<100; ++i)
//			a.observe(i, Math.random()<0.25?Math.random()*100:i);
//	}
}
