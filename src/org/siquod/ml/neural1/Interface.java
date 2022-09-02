package org.siquod.ml.neural1;

import java.util.HashSet;

public final class Interface {
	public int offset=-1;
	public int count;
	public String name;
	HashSet<String> dontCompute=new HashSet<>();
	public TensorFormat tf;
	int lifeIndex=-1;

	public Interface(TensorFormat tf){
		this(null, tf.count(), tf);
	}
	public Interface(String name, TensorFormat tf){
		this(name, tf.count(), tf);
	}

	public Interface(int count, TensorFormat tf){
		this(null, count, tf);
	}
	public Interface(String name, int count, TensorFormat tf){
		this.name=name;
		this.count=count;
		this.tf=tf==null?new TensorFormat(count):tf;
	}

	public Interface tensorFormat(TensorFormat tf){
		this.tf=tf;
		return this;
	}
	public void dontComputeInPhase(String phase) {
		dontCompute.add(phase);
	}
	public boolean shouldCompute(String phase){
		return !dontCompute.contains(phase);
	}
//	public double get(ActivationSet as, int[] pos) {
//		return tf.get(as, this, pos);
//	}
//	public void add(ActivationSet as, int[] pos, double value) {
//		tf.add(as, this, pos, value);
//	}
	public float get(ActivationSet as, int[] pos, int c) {
		return tf.get(as, this, pos, c);
	}
	public void add(ActivationSet as, int[] pos, int c, float value) {
		tf.add(as, this, pos, c, value);
	}
	public int channels() {
		return tf.channels();
	}

	
}