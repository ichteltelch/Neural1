package org.siquod.ml.neural1.modules.regularizer;

import org.siquod.ml.neural1.ParamSet;

public abstract class Regularizer {
	float strength=1;
	public Regularizer(double s) {
		strength=(float)s;
	}
	final public void regularize(
			ParamSet weights, ParamSet gradients,
			int o, int n1, int s1,
			int n2, int s2,
			float globalRegularization)
	{
		float eff = globalRegularization*strength;
		for(int i=0; i<n1; ++i){
			regularize(weights, gradients, o + i*s1, n2, s2, eff);
		}
	}
	public abstract void regularize(ParamSet weights, ParamSet gradients, int o, int n, int s, float eff) ;
}
