package org.siquod.ml.neural1.neurons;

public class FadeInNeuron extends ParameterizedNeuron{
	public final Neuron back;
	public FadeInNeuron(Neuron back) {
		this.back=back;
	}
	@Override
	public float f(float x, float a) {
		return a * back.f(x) + (1-a) * x;
	}
	@Override
	public float dfdx(float x, float a) {
		return a * back.df(x) + (1-a);
	}
	@Override
	public float dfda(float x, float a) {
		return back.f(x) - x;
	}

}
