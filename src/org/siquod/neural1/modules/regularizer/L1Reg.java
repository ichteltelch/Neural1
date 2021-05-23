package org.siquod.neural1.modules.regularizer;

import org.siquod.neural1.ParamSet;

public class L1Reg extends Regularizer{
	public L1Reg(double s){
		super(s);
	}
	@Override
	public void regularize(ParamSet weights, ParamSet gradients, int o, int n, int s, float eff) {
		for(int i=0; i<n; ++i){
			int index = o+i*s;
			gradients.add(index, eff*Math.signum(weights.get(index)));
		}
	}
}
