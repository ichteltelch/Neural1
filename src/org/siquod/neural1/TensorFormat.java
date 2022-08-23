package org.siquod.neural1;

import java.util.Arrays;

public final class TensorFormat {
	public final int rank;
	public final int[] dims;
	public TensorFormat(int... dims){
		this.dims = dims;
		this.rank = dims.length;
	}
	public int index(int i0){
		if(i0<0 || i0>=dims[0])
			return -1;
		return i0;
	}
	public int index(int i0, int i1){
		if(i0<0 || i0>=dims[0])
			return -1;
		if(i1<0 || i1>=dims[1])
			return -1;
		return i0 + dims[0]*i1;
	}
	public TensorFormat flattenIndexAndNext(int firstIndex) {
		if(firstIndex<0)
			throw new IllegalArgumentException("index must be positive");
		if(firstIndex>=rank)
			throw new IllegalArgumentException("index must be less than rank");
		if(firstIndex==rank-1)
			throw new IllegalArgumentException("index must not be the last one");
		int[] ndims = new int[rank-1];
		for(int i=0; i<firstIndex; ++i)
			ndims[i] = dims[i];
		ndims[firstIndex] = dims[firstIndex]*dims[firstIndex+1];
		for(int i=firstIndex+1; i<rank; ++i)
			ndims[i-1] = dims[i];
		return new TensorFormat(ndims);
	}
	public TensorFormat insertUnitIndex(int atIndex) {
		if(atIndex<0)
			throw new IllegalArgumentException("index must be positive");
		if(atIndex>rank)
			throw new IllegalArgumentException("index must not be greater than rank");
		int[] ndims = new int[rank+1];
		for(int i=0; i<atIndex; ++i)
			ndims[i] = dims[i];
		ndims[atIndex] = 1;
		for(int i=0; i<atIndex; ++i)
			ndims[i+1] = dims[i];
		return new TensorFormat(ndims);

	}
	public int index(int i0, int i1, int i2){
		if(i0<0 || i0>=dims[0])
			return -1;
		if(i1<0 || i1>=dims[1])
			return -1;
		if(i2<0 || i2>=dims[2])
			return -1;
		return i0 + dims[0]*(i1 + dims[1]*i2);
	}
//	public int index(int[] i){
//		for(int d=0; d<dims.length; ++d) {
//			if(i[d]<0 || i[d]>=dims[d])
//				return -1;
//		}
//		int r=i[dims.length-1];
//		for(int d=dims.length-2; d>=0; --d) {
//			r = i[d] + r * (dims[d]);
//		}
//		return r;
//	}
	public int index(int[] i, int c){
		for(int d=0; d<dims.length; ++d) {
			int iv = d==dims.length-1?c:i[d];
			if(iv<0 || iv>=dims[d])
				return -1;
		}
		int r=c;
		for(int d=dims.length-2; d>=0; --d) {
			r = i[d] + r * (dims[d]);
		}
		return r;
	}

	public float get(ActivationSet a, Interface i, int i0){
		if(i0<0 || i0>=dims[0])
			return 0;
		return a.get(i, i0);
	}
	public float get(ActivationSet a, Interface i, int i0, int i1){
		if(i0<0 || i0>=dims[0])
			return 0;
		if(i1<0 || i1>=dims[1])
			return 0;
		return a.get(i, i0 + dims[0]*i1);
	}
	public float get(ActivationSet a, Interface i, int i0, int i1, int i2){
		if(i0<0 || i0>=dims[0])
			return 0;
		if(i1<0 || i1>=dims[1])
			return 0;
		if(i2<0 || i2>=dims[2])
			return 0;
		return a.get(i, i0 + dims[0]*(i1 + dims[1]*i2));
	}
//	public double get(ActivationSet a, Interface in, int[] i){
//		for(int d=0; d<dims.length; ++d) {
//			if(i[d]<0 || i[d]>=dims[d])
//				return -1;
//		}
//		int r=i[dims.length-1];
//		for(int d=dims.length-2; d>=0; --d) {
//			r = i[d] + r * (dims[d]);
//		}
//		return a.get(in, r);
//	}
	public float get(ActivationSet a, Interface in, int[] i, int c){
		for(int d=0; d<dims.length; ++d) {
			int iv = d==dims.length-1?c:i[d];
			if(iv<0 || iv>=dims[d])
				return -1;
		}
		int r=c;
		for(int d=dims.length-2; d>=0; --d) {
			r = i[d] + r * (dims[d]);
		}
		return a.get(in, r);
	}
	public void add(ActivationSet a, Interface in, int[] i, int c, float val){
		for(int d=0; d<dims.length; ++d) {
			if(i[d]<0 || i[d]>=dims[d])
				return;
		}
		int r=c;
		for(int d=dims.length-2; d>=0; --d) {
			r = i[d] + r * (dims[d]);
		}
		a.add(in, r, val);
	}
	public void set(ActivationSet a, Interface in, int[] i, int c, float val){
		for(int d=0; d<dims.length; ++d) {
			if(i[d]<0 || i[d]>=dims[d])
				return;
		}
		int r=c;
		for(int d=dims.length-2; d>=0; --d) {
			r = i[d] + r * (dims[d]);
		}
		a.set(in, r, val);
	}
//	public void add(ActivationSet a, Interface in, int[] i, double val){
//		for(int d=0; d<dims.length; ++d) {
//			if(i[d]<0 || i[d]>=dims[d])
//				return;
//		}
//		int r=i[dims.length-1];
//		for(int d=dims.length-2; d>=0; --d) {
//			r = i[d] + r * (dims[d]);
//		}
//		a.add(in, r, val);
//	}
	public void add(ActivationSet a, Interface i, int i0, float val){
		if(i0<0 || i0>=dims[0])
			return;
		a.add(i, i0, val);
	}
	public void add(ActivationSet a, Interface i, int i0, int i1, float val){
		if(i0<0 || i0>=dims[0])
			return;
		if(i1<0 || i1>=dims[1])
			return;
		a.add(i, i0 + dims[0]*i1, val);
	}
	public void add(ActivationSet a, Interface i, int i0, int i1, int i2, float val){
		if(i0<0 || i0>=dims[0])
			return;
		if(i1<0 || i1>=dims[1])
			return;
		if(i2<0 || i2>=dims[2])
			return;
		a.add(i, i0 + dims[0]*(i1 + dims[1]*i2), val);
	}
	public int channels() {
		return dims[dims.length-1];
	}
	public int count() {
		int r=1;
		for(int d: dims)
			r*=d;
		return r;
	}
	public int channelStride() {
		int r=1;
		for(int i=0; i<dims.length-1; ++i)
			r*=dims[i];
		return r;
	}
	public TensorFormat withChannels(int i) {
		int[] ni=dims.clone();
		ni[ni.length-1]=i;
		return new TensorFormat(ni);
	}
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		if (o == null)
			return false;
		if (o instanceof TensorFormat) {
			TensorFormat a = (TensorFormat) o;
			return Arrays.equals(dims, a.dims);
		}
		return false;
	}
	@Override
	public int hashCode() {
		return Arrays.hashCode(dims);
	}
	
}
