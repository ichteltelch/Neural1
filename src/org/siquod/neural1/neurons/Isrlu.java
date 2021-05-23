package org.siquod.neural1.neurons;


public class Isrlu extends ParameterizedNeuron{
	public float f(float x, float a){
		return x>=0?x:(x/(float)Math.sqrt(1+a*x*x));
	}

	@Override
	public float dfdx(float x, float a) {
		if(x>=0)
			return 1;
		float r = 1/(1+a*x*x);
		float c = (float)Math.sqrt(r);
		return c*r;
	}

	@Override
	public float dfda(float x, float a) {
		if(x>=0)
			return 0;
		float q=x*x;
		float r = 1+a*q;
		return q*x*0.5f/(r*(float)Math.sqrt(r));
	}

}