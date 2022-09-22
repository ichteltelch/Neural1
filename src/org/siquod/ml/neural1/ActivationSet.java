package org.siquod.ml.neural1;

import java.util.Arrays;

public final class ActivationSet implements Cloneable{
	private float[] value;
	private int[] decisions;
	public ActivationSet(int size, int decCount){
		value=new float[size];
		decisions=new int[decCount];
	}
	public void clear(){
		Arrays.fill(value, 0);
	}
	public void add(Interface i, int index, float d){
		assert index>=0;
		assert index<i.count;
//		if(Double.isNaN(d))
//			throw new IllegalArgumentException();
//		if(Double.isInfinite(d))
//			throw new IllegalArgumentException();
//		if(Double.isInfinite(value[i.start + index] +d))
//			throw new IllegalArgumentException();
		value[i.offset + index] += d;
	}
	public void set(Interface i, int index, float d){
		assert index>=0;
		assert index<i.count;
//		if(Double.isNaN(d))
//			throw new IllegalArgumentException();
//		if(Double.isInfinite(d))
//			throw new IllegalArgumentException();
//		if(Double.isInfinite(value[i.start + index] +d))
//			throw new IllegalArgumentException();
		value[i.offset + index] = (float) d;
	}
	public void mult(Interface i, int index, float d) {
		assert index>=0;
		assert index<i.count;
		value[i.offset + index] *= d;
	}
	public float get(Interface i, int index){
		assert index>=0;
		assert index<i.count;
		float ret = value[i.offset + index];
		if(Double.isNaN(ret))
			System.out.println();
		return ret;
	}

	public void clear(Interface i){
		Arrays.fill(value, i.offset, i.offset+i.count, 0);
	}
	public void clearAllChannels(Interface i, int[] inst) {
		int o = i.offset+i.tf.index(inst, 0);
		int s=i.tf.channelStride();
		for(int j=i.channels()-1; j>=0; --j) {
			value[o+j*s]=0;
		}
	}
	public void set(Interface in, float[] input) {
		System.arraycopy(input, 0, value, in.offset, in.count);	
	}
	public void get(Interface in, float[] output) {
		System.arraycopy(value, in.offset, output, 0, in.count);	
	}
	public void add(Interface in, float[] addThis) {
		for(int i=0; i<in.count; ++i)
			value[in.offset+i]+=addThis[i];
	}
	public void setDecision(int i, int d) {
		if(i==-1)
			return;
		decisions[i]=d;
	}
	public int getDecision(int i) {
		return decisions[i];
	}
//	public double get(Interface in, int[] i) {
//		return in.tf.get(this, in, i);
//	}
//	public void add(Interface out, int[] i, double val) {
//		out.tf.add(this, out, i, val);
//	}
	public float get(Interface in, int[] i, int c) {
		return in.tf.get(this, in, i, c);
	}
	public void add(Interface out, int[] i, int c, float val) {
		out.tf.add(this, out, i, c, val);
	}
	public void set(Interface out, int[] i, int c, float val) {
		out.tf.set(this, out, i, c, val);
	}
	@Override
	public ActivationSet clone(){
		try {
			ActivationSet r = (ActivationSet) super.clone();
			r.value=value.clone();
			r.decisions=decisions.clone();
			return r;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			return null;
		}
	}
}
