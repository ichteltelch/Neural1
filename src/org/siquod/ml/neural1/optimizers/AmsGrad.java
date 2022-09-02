package org.siquod.ml.neural1.optimizers;

import org.siquod.ml.neural1.ParamSet;

public class AmsGrad extends Updater{
	public float beta1=0.9f;
	public float beta2=0.999f;
	public float epsilon=1e-8f;
	float beta1pow=1;
	float beta2pow=1;
	ParamSet m, v, vm;
	public void apply(ParamSet ps, ParamSet grad, float lr, float totalWeight){
		beta1pow*=beta1;
		beta2pow*=beta2;
		if(m==null){
			m=new ParamSet(ps.size());
			v=new ParamSet(ps.size());
		}
		if(vm==null) {
			vm=new ParamSet(ps.size());
		}
		ps.amsGrad(lr, beta1, beta2, epsilon, beta1pow, beta2pow, grad, m ,v, vm, totalWeight);
	}
	@Override
	protected void cloneData() {
		m=m.clone();
		v=v.clone();
		vm=vm.clone();
	}
	public void forgetVMax() {
		vm=null;
	}
}
