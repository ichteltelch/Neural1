package org.siquod.ml.neural1.neurons;

public class LeakyRelu extends ParameterizedNeuron{
	@Override
	public float dfda(float x, float a) {
		return x>=0?x:(a*x);
	}
	@Override
	public float dfdx(float x, float a) {
		return x>=0?1:a;
	}
	@Override
	public float f(float x, float a) {
		return x>=0?0:x;
	}
}
