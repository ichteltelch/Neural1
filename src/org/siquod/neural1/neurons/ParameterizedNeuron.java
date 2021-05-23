package org.siquod.neural1.neurons;

public abstract class ParameterizedNeuron {
	public abstract float f(float x, float a);
	public abstract float dfdx(float x, float a);
	public abstract float dfda(float x, float a);
	public Neuron fixA(final float a){
		return new Neuron() {			
			@Override
			public float f(float x) {
				return ParameterizedNeuron.this.f(x, a);
			}
			
			@Override
			public float df(float x) {
				return ParameterizedNeuron.this.dfdx(x, a);
			}
		};
	}
}
