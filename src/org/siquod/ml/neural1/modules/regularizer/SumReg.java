package org.siquod.ml.neural1.modules.regularizer;

import org.siquod.ml.neural1.ParamSet;

public class SumReg extends Regularizer{
	public SumReg(Regularizer ... rs) {
		super(1);
		sub=rs.clone();
	}
	Regularizer[] sub;
	@Override
	public void regularize(ParamSet weights, ParamSet gradients, int o, int n, int s, float eff) {
		for(Regularizer r: sub)
			r.regularize(weights, gradients, o, n, s, eff*r.strength);
	}
}
