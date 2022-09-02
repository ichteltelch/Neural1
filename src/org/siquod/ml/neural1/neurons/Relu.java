package org.siquod.ml.neural1.neurons;

public class Relu extends Neuron{

	@Override
	public float f(float in) {
		return Math.max(in, 0);
	}

	@Override
	public float df(float in) {
		return (Math.signum(in)+1)*0.5f;
	}

}
