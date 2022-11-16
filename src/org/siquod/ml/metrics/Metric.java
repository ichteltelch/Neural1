package org.siquod.ml.metrics;

public class Metric {
	public static final Metric ACCURACY = new Metric("Accuracy", true);
	public static final Metric LOSS = new Metric("Loss", false);
	public static final Metric TRAIN_ACCURACY = new Metric("Train accuracy", true);
	public static final Metric TRAIN_LOSS = new Metric("Train loss", false);
	public final String name;
	public final boolean biggerIsBetter;
	public Metric(String name, boolean biggerIsBetter) {
		this.name = name;
		this.biggerIsBetter=biggerIsBetter;
	}
	public String toString() {
		return name;
	}
	public Metric movingAverage(int past, int future) {
		return new Metric("MA("+past+","+future+") of "+name, biggerIsBetter);
	}
	public Metric movingMedian(int past, int future) {
		return new Metric("MM("+past+","+future+") of "+name, biggerIsBetter);
	}
	public Metric movingRobustAverage(int past, int future, double fraction) {
		return new Metric("MRA("+past+","+future+","+fraction+") of "+name, biggerIsBetter);
	}
}
