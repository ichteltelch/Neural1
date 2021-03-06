package org.siquod.neural1.updaters;

import org.siquod.neural1.ParamSet;

public class Rprop extends Updater{

	ParamSet lastGrad, gamma;
	public void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight){
		if(lastGrad==null){
			lastGrad=new ParamSet(ps.size());
			gamma=new ParamSet(ps.size());
		}
		ps.rprop(grad, lastGrad, gamma, lr);
	}
	@Override
	protected void cloneData() {
		lastGrad=lastGrad==null?null:lastGrad.clone();
		gamma=gamma==null?null:gamma.clone();
	}
}
