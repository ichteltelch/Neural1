package org.siquod.neural1.neurons;

public class SeLU extends Neuron {
	float lambda, alpha;
	public SeLU() {
		this(1.6732, 1.0507);
	}
	public SeLU(double a, double l) {
		lambda=(float) l;
		alpha=(float) a;
	}
	@Override
	public float f(float x) {
		return lambda * (x>=0?x:(alpha*((float)Math.exp(x)-1)));
	}
	@Override
	public float df(float x) {
		return lambda * (x>=0?1:(alpha*(float)Math.exp(x)));
	}
}
