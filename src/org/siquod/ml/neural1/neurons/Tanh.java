package org.siquod.ml.neural1.neurons;

public class Tanh extends Neuron{
	public static final Neuron INST = new Tanh();

	@Override
	public float f(float x) {
		return (float) Math.tanh(x);
	}

	@Override
	public float df(float x) {
		float fx=(float)Math.tanh(x);
		return 1-fx*fx;
	}

}
