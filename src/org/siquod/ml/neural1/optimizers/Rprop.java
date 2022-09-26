package org.siquod.ml.neural1.optimizers;

import org.siquod.ml.neural1.ParamSet;

public class Rprop extends Updater{

	ParamSet lastGrad, gamma;
	public void apply(ParamSet ps, ParamSet grad, ParamSet lrMult, float lr, float totalWeight){
		if(lastGrad==null){
			lastGrad=new ParamSet(ps.size());
			gamma=new ParamSet(ps.size());
		}
		ps.rprop(grad, lastGrad, gamma, lr, lrMult);
	}
	@Override
	protected void cloneData() {
		lastGrad=lastGrad==null?null:lastGrad.clone();
		gamma=gamma==null?null:gamma.clone();
	}
}
