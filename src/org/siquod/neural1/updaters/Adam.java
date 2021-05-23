package org.siquod.neural1.updaters;

import org.siquod.neural1.ParamSet;

public class Adam extends Updater{
	public float beta1=0.9f;
	public float beta2=0.999f;
	public float epsilon=1e-8f;
	float beta1pow=1;
	float beta2pow=1;
	ParamSet m, v;
	public void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight){
		beta1pow*=beta1;
		beta2pow*=beta2;
		if(m==null){
			m=new ParamSet(ps.size());
			v=new ParamSet(ps.size());
		}
		ps.adam(lr, beta1, beta2, epsilon, beta1pow, beta2pow, grad, m ,v, totalWeight);
	}
	@Override
	protected void cloneData() {
		m=m.clone();
		v=v.clone();
	}
}
