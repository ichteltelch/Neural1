package org.siquod.ml.neural1.optimizers;

import org.siquod.ml.neural1.ParamSet;

public class SGD extends Updater{

	@Override
	public void apply(ParamSet ps, ParamSet grad, ParamSet lrMult, float lr, float totalWeight) {
		float[] psv = ps.value;
		float[] grv = grad.value;
		float[] lrv = lrMult.value;
		for(int i=0; i<psv.length; ++i)
			psv[i] -= grv[i]*lrv[i]*lr/totalWeight;
		
	}
	@Override
	protected void cloneData() {
	}
}
