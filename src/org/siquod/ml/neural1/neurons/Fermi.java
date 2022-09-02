package org.siquod.ml.neural1.neurons;

public class Fermi extends Neuron{

	public static final Neuron INST = new Fermi();

	@Override
	public float f(float x) {
		return 1/(1+(float)Math.exp(-x));
	}

	@Override
	public float df(float x) {
		float ex = (float)Math.exp(x);
		float ex1=1+ex;
		return ex/(ex1*ex1);
	}
	
}
