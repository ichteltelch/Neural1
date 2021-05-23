package org.siquod.neural1.updaters;

import org.siquod.neural1.ParamSet;

public class SGD extends Updater{

	@Override
	public void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight) {
		ps.addMultiple(grad, -lr/totalWeight);
		
	}
	@Override
	protected void cloneData() {
	}
}
