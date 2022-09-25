package org.siquod.ml.neural1.modules.regularizer;

import org.siquod.ml.neural1.ParamSet;

public class L2UnitReg extends Regularizer{
	public L2UnitReg(double s){
		super(s);
	}
	@Override
	public void regularize(ParamSet weights, ParamSet gradients, int o, int n, int s, float eff) {
		double lq = 0;
		for(int i=0; i<n; ++i) {
			int index = o+i*s;
			double wt = weights.get(index);
			lq += wt*wt;
		}
		double l = Math.sqrt(lq);
		double le = l-1;
		double loss = le*le;
		double ðloss = eff;
		double ðle = ðloss*2*le;
		double ðl = ðle;
		double ðlq = ðl * 0.5/l;
		for(int i=0; i<n; ++i) {
			int index = o+i*s;
			double wt = weights.get(index);
			double ðwt = ðlq*2*wt;
			gradients.add(index, ðwt);
		}
	}
	@Override
	public String toString() {
		return "L2("+strength+")-1";
	}
}
