package org.siquod.neural1.neurons;

public abstract class Neuron extends ParameterizedNeuron{
	@Override
	public final float f(float x, float y) {
		return f(x);
	}
	@Override
	public final float dfdx(float x, float y) {
		return df(x);
	}
	@Override
	public final float dfda(float x, float y) {
		return 0;
	}
	public abstract float f(float x);
	public abstract float df(float x);
}
