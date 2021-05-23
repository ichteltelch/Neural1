package org.siquod.neural1.updaters;

import org.siquod.neural1.ParamSet;

public abstract class Updater implements Cloneable{
	public abstract void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight);
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
