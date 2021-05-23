package org.siquod.neural1;

public class ActivationBatch implements Cloneable{
	public ActivationSeq[] a;
	public ActivationSeq batchParams;
	public int length;
	public ActivationBatch(int batchSize, int time, InterfaceAllocator activ, InterfaceAllocator bparam) {
		length=batchSize;
		a=new ActivationSeq[batchSize];
		for(int i=0; i<a.length; ++i)
			a[i]=activ.makeSeq(time).init().entire();
		batchParams=bparam.makeSeq(time).init().entire();
	}
	private ActivationBatch(ActivationSeq[] a, ActivationSeq b) {
		this.a=a.clone();
		this.batchParams=b;
		length=a.length;
	}
	public ActivationBatch shallowClone() {
		return new ActivationBatch(a, batchParams);
	}
	public void clear() {
		for(ActivationSeq as: a)
			for(ActivationSet at: as.steps)
				at.clear();
	}
	@Override
	public ActivationBatch clone(){
		try {
			ActivationBatch r = (ActivationBatch) super.clone();
			r.a=a.clone();
			for(int i=0; i<a.length; ++i)
				r.a[i]=a[i].clone();
			r.batchParams=batchParams.clone();
			return r;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			return null;
		}
	}
}
 