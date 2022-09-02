package org.siquod.ml.neural1.optimizers;

import org.siquod.ml.neural1.ParamSet;

public class Adam extends Updater{
	public float beta1=0.9f;
	public float beta2=0.999f;
	public float epsilon=1e-8f;
	float beta1exp=1;
	float beta2exp=1;
	ParamSet m, v;
	public void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight){
		beta1exp*=beta1;
		beta2exp*=beta2;
		if(m==null){
			m=new ParamSet(ps.size());
			v=new ParamSet(ps.size());
		}
		ps.adam(lr, beta1, beta2, epsilon, beta1exp, beta2exp, grad, m ,v, totalWeight);
	}
	@Override
	protected void cloneData() {
		m=m==null?null:m.clone();
		v=v==null?null:v.clone();
	}
	@Override
	public String toString() {
		return "Adam(beta1="+beta1+", beta2="+beta2+", epsilon="+epsilon+", beta1exp="+beta1exp+", beta2exp="+beta2exp+")";
	}
}
