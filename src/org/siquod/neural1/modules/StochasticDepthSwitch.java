package org.siquod.neural1.modules;

import java.util.Collections;
import java.util.List;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ActivationSeq;
import org.siquod.neural1.ActivationSet;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;

/**
 * Note: A stochastic depth module may not contain a batch normalization Layer!
 * @author bb
 *
 */
public class StochasticDepthSwitch implements InOutModule{
	public float keepProbability;
	public int dropoutOffset=-1;
	Interface out;
	InOutModule inner;
	public StochasticDepthSwitch(InOutModule i, double p) {
		keepProbability=(float) p;
		inner = i;
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		inner.allocate(ia);
		if(dropoutOffset==-1)
			dropoutOffset=ia.allocateDropout(1);
		out=inner.getOut();
	}
	@Override
	public void allocate(ParamAllocator ia) {
		inner.allocate(ia);
	}
	@Override
	public void share(ParamBlocks ps) {
		inner.share(ps);
	}
	@Override
	public ParamBlocks getParamBlocks() {
		return inner.getParamBlocks();
	}
	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(training==ForwardPhase.TRAINING){
//			Iterable<ActivationSeq> nas=() -> new Iterator<ActivationSeq>() {
//				Iterator<ActivationSeq> back=as.a.iterator();
//				@Override
//				public boolean hasNext() {
//					return back.hasNext();
//				}
//
//				@Override
//				public ActivationSeq next() {
//					ActivationSeq r=back.next();
//					if(r.getDropout(dropoutOffset)==0)
//						return null;
//					return r;
//				}
//			};
			as=as.shallowClone();
			for(int b=0; b<as.length; ++b) {
				if(as.a[b]==null) continue;
				if(as.a[b].getDropout(dropoutOffset)==0) {
					as.a[b]=null;
				}
			}
			inner.forward(training, params, as, t, inst);
		}else {
			inner.forward(training, params, as, t, inst);
			int outcount = out.channels();
			for(ActivationSeq b: as.a) {
				ActivationSet a=b.get(t);
				for(int i=0; i<outcount; ++i){
					a.mult(out, out.tf.index(inst, i), keepProbability);
				}
			}
		}
	}
	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		errors=errors.shallowClone();
		for(int b=0; b<errors.length; ++b) {
			if(errors.a[b]==null)continue;
			if(errors.a[b].getDropout(dropoutOffset)==0) {
				errors.a[b]=null;
			}
		}
		inner.backprop(phase, params, as, errors, t, inst);
	}

	@Override
	public void dontComputeInPhase(String phase) {
	}

	@Override
	public Interface getIn() {
		return inner.getIn();
	}
	@Override
	public Interface getOut() {
		return inner.getOut();
	}
	//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}

//	@Override
//	public void initializeBatch(ActivationSeq as) {
//		inner.initializeBatch(as);
//	}
	@Override
	public void initializeRun(ActivationBatch as, boolean training) {
		if(training)
			for(ActivationSeq a: as.a)
				a.sampleDropout(dropoutOffset, 1, keepProbability);
		inner.initializeRun(as, training);
	}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		errors=errors.shallowClone();
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null)continue;
			if(errors.a[b].getDropout(dropoutOffset)==0) {
				errors.a[b]=null;
			}
		}
		inner.gradients(phase, params, as, errors, gradients, t, inst);
	}
	@Override
	public int dt() {
		return inner.dt();
	}
	@Override
	public int[] shift() {
		return inner.shift();
	}
	@Override
	public List<Module> getSubmodules() {
		return Collections.singletonList(inner);
	}
}
