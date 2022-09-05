package org.siquod.ml.neural1;

import java.util.Arrays;


public final class ParamSet implements Cloneable{
	private float[] value;
	public ParamSet(int size){
		value=new float[size];
	}
	public ParamSet(float[] v){
		value=v.clone();
	}
	public void set(ParamBlock i, int index, double d){
		assert index>=0;
		assert index<i.count;
		value[i.start + index] = (float)d;
	}
	public void add(ParamBlock i, int index, double d){
		assert index>=0;
		assert index<i.count;
		value[i.start + index] += d;
//		if(Double.isInfinite(i.start + value[index]))
//			throw new IllegalArgumentException();
	}
	public float get(ParamBlock i, int index){
		assert index>=0;
		assert index<i.count;
//		if(13352==i.start + index)
//			System.out.println();
		return value[i.start + index];
	}
	public float get(int index) {
//		if(13352== index)
//			System.out.println();
		return value[index];
	}
	public void add(int index, double d) {
//		if(Double.isNaN(d))
//			throw new IllegalArgumentException();
//		if(Double.isInfinite((float)d))
//			throw new IllegalArgumentException();
		value[index] += d;
//		if(Double.isInfinite(value[index]))
//			throw new IllegalArgumentException();
	}
	public void set(int index, double d) {
//		if(Double.isNaN(d))
//			throw new IllegalArgumentException();
		value[index] = (float)d;
//		if(Double.isInfinite(value[index]))
//			throw new IllegalArgumentException();
	}
	public int size() {
		return value.length;
	}
	public void clear() {
		Arrays.fill(value, 0);
	}
	public void clear(ParamBlock b) {
		Arrays.fill(value, b.start, b.start+b.count, 0);
	}
	public void addMultiple(ParamSet ps, double factor){
		float f=(float)factor;
		float[] oval=ps.value;
		for(int i=value.length-1; i>=0; --i)
			value[i] += oval[i]*f;
	}
	public ParamSet clone(){
		return new ParamSet(value);
	}
	public void rprop(ParamSet grad, ParamSet lastGrad, ParamSet gamma, float f) {
		float[] gr  = grad.value;
		float[] og = lastGrad.value;
		float[] ga = gamma.value;
		for(int i=value.length-1; i>=0; --i){
			float gv = gr[i];
			float ogv=og[i];
			float sign=gv*ogv;
			float gav=ga[i];
			if(gav==0)
				gav=0.1f*f;
			if(sign>0){
				gav=Math.min(gav*1.2f, 50);
			}else if(sign<0){
				gav=Math.max(gav*0.5f, 1e-6f);
				ogv=Float.NaN;
			}
			value[i] -= Math.signum(gv)*gav;
//			if(Double.isInfinite(value[i]))
//				throw new IllegalArgumentException();
			ga[i] = gav;
		}
		System.arraycopy(gr, 0, og, 0, gr.length);
	}
	public void adam(float lr, float beta1, float beta2, float epsilon, 
			float beta1pow, float beta2pow, ParamSet grad, ParamSet m, ParamSet v, float totalWeight) {
		float[] gr  = grad.value;
		float[] mv = m.value;
		float[] vv = v.value;
		for(int i=value.length-1; i>=0; --i){
			float grv=gr[i]/totalWeight;
			float mn = (mv[i] = beta1*mv[i] + (1-beta1)*grv)/(1-beta1pow);
			float vn = (vv[i] = beta2*vv[i] + (1-beta2)*grv*grv)/(1-beta2pow);
			value[i] -= lr*mn/(Math.sqrt(vn)+epsilon);
			
		}
	
	}
	public void amsGrad(float lr, float beta1, float beta2, float epsilon, 
			float beta1pow, float beta2pow, ParamSet grad, ParamSet m, ParamSet v, ParamSet vm, float totalWeight) {
		float[] gr  = grad.value;
		float[] mv = m.value;
		float[] vv = v.value;
		float[] vmv = vm.value;
		for(int i=value.length-1; i>=0; --i){
			float grv=gr[i]/totalWeight;
			float mn = (mv[i] = beta1*mv[i] + (1-beta1)*grv)/(1-beta1pow);
			float vn = (vv[i] = beta2*vv[i] + (1-beta2)*grv*grv)/(1-beta2pow);
			float vmn=vmv[i]=beta2pow>0.97?vn:Math.max(vn, vmv[i]);
			value[i] -= lr*mn/(Math.sqrt(vmn)+epsilon);
		}
	
	}
	public void clip(float d) {
		for(int i=0; i<value.length; ++i) {
			float v=value[i];
			v = Math.max(v, -d);
			v=Math.min(v, d);
			value[i]=v;
		}
	}
	public double dot(ParamSet other) {
		double r=0;
		for(int i=0; i<value.length; i++)
			r+=value[i]*other.value[i];
		return r;
	}

	public void mult(float f) {
		for(int i=0; i<value.length; ++i)
			value[i]*=f;
	}
	public void set(float[] params) {
		if(params.length!=value.length)
			throw new IllegalArgumentException("Cannot set parameters: length mismatch");
		System.arraycopy(params, 0, value, 0, params.length);
	}
}
