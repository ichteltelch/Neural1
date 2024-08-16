package org.siquod.ml.neural1.modules;

import java.util.Collections;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ActivationSeq;
import org.siquod.ml.neural1.ActivationSet;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;

public class Maxout implements InOutModule{
	Interface in;
	Interface out;
	int k;
	int di;
	public Maxout(Maxout copyThis) {
		this.in=copyThis.in;
        this.out=copyThis.out;
        this.k=copyThis.k;
        this.di=copyThis.di;
	}
	@Override
	public Maxout copy() {
		return this;
	}
	public Maxout() {
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		k = in.count/out.count;
		if(out.count*k!=in.count){
			throw new IllegalArgumentException("input interface size must be a multiple of output interface size");
		}
		if(in.tf!=null){
			if(in.tf.dims[in.tf.dims.length-1]%k != 0){
				throw new IllegalArgumentException("last dimension of input interface size not a multiple of k");
			}
			if(out.tf==null){
				int[] dims=in.tf.dims.clone();
				dims[dims.length-1]/=k;
				out.tf=new TensorFormat(dims);
			}
		}
		di = ia.allocateDecision(out.count);
	}

	@Override
	public void allocate(ParamAllocator ia) {
	}

	@Override
	public void share(ParamBlocks ps) {		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		return null;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(inst==null) {
			int outcount=out.count;
			for(ActivationSeq b: as) {
				if(b==null) continue;
				ActivationSet a=b.get(t);
				for(int o=0; o<outcount; ++o){
					int mi=-1;
					float max=Float.NEGATIVE_INFINITY;
					for(int i=0; i<k; ++i){
						float v=a.get(in, o+i*out.count);
						if(v>max){
							max=v;
							mi=i;
						}
					}
					a.add(out, o, max);
					a.setDecision(di+out.tf.index(o), mi);
				}
			}
		}else {
			int outcount=out.channels();
			int[] pos=inst;
			for(ActivationSeq b: as) {
				if(b==null) continue;
				ActivationSet a=b.get(t);
				for(int o=0; o<outcount; ++o){
					int mi=-1;
					float max=Float.NEGATIVE_INFINITY;
					for(int i=0; i<k; ++i){
						float v=a.get(in, pos, o+i*out.count);
						if(v>max){
							max=v;
							mi=i;
						}
					}
					a.add(out, pos, o, max);
					a.setDecision(di+out.tf.index(pos, o), mi);
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst==null) {
			int outcount = out.count;
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int o=0; o<outcount; ++o){
					int i = a.getDecision(di+out.tf.index(o));
					if(i>=0)
						e.add(in, o+i*outcount, e.get(out, o));
				}		
			}		
		}else {
			int outcount = out.channels();
			int[] pos=inst;
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int o=0; o<outcount; ++o){
					int i = a.getDecision(di+out.tf.index(pos, o));
					if(i>=0)
						e.add(in, pos, o+i*outcount, e.get(out, pos, o));
				}		
			}
		}
	}

//	//	@Override
//	public void declareDependencies(Dependencies d) {
//		d.declare(new InputDependency(in, this, 0));
//		d.declare(new OutputDependency(this, out));
//	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}

	//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}
	@Override
	public Interface getIn() {
		return in;
	}
	@Override
	public Interface getOut() {
		return out;
	}
	@Override
	public int dt() {
		return 0;
	}
	@Override
	public int[] shift() {
		return null;
	}
	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
}
