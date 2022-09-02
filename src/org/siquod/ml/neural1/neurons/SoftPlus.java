package org.siquod.ml.neural1.neurons;

public class SoftPlus extends Neuron{

	@Override
	public float f(float x) {
		return (float) Math.log(1+Math.exp(x));
	}

	@Override
	public float df(float x) {
		return 1/(1+(float)Math.exp(-x));
	}
	
}
