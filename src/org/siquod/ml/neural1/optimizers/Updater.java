package org.siquod.ml.neural1.optimizers;

import org.siquod.ml.neural1.ParamSet;

public abstract class Updater implements Cloneable{
	public abstract void apply(ParamSet ps, ParamSet grad, ParamSet lrMult, float lr, float totalWeight);
	@Override
	public Updater clone() {
		try {
			Updater r = (Updater) super.clone();
			r.cloneData();
			return r;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			return null;
		}
	}
	protected abstract void cloneData();
}
