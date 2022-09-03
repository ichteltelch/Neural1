package org.siquod.ml.neural1.neurons;

public class CrossFadeNeuron extends ParameterizedNeuron{
	public final Neuron at0, at1;
	public CrossFadeNeuron(Neuron at0, Neuron at1) {
		this.at0 = at0;
		this.at1 = at1;
	}
	@Override
	public float f(float x, float a) {
		return a * at1.f(x) + (1-a) * at0.f(x);
	}
	@Override
	public float dfdx(float x, float a) {
		return a * at1.df(x) + (1-a) * at0.df(x);
	}
	@Override
	public float dfda(float x, float a) {
		return at1.f(x) - at0.f(x);
	}
	
}
