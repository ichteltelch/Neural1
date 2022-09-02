package org.siquod.ml.neural1.optimizers;

import org.siquod.ml.neural1.ParamSet;

public class SGD extends Updater{

	@Override
	public void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight) {
		ps.addMultiple(grad, -lr/totalWeight);
		
	}
	@Override
	protected void cloneData() {
	}
}
