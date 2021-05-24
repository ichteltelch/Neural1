package org.siquod.neural1.modules.loss;

import java.util.Arrays;
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
 * This negative log-likelihood loss layer expects a logarithmized probability distribution in 
 * its input interface "in", a probability distribution in its input interface "target" and 
 * outputs the loss in its one-dimensional output interface "loss". 
 * @author bb
 *
 */
public class NllLoss extends LossLayer{
	Interface in, target, loss;
	List<String> phases;
	public NllLoss(String... ph) {
		phases=Arrays.asList(ph.clone());
	}


	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		target=ia.get("target", in.count);
		loss=ia.get("loss", 1);
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
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch ass, int t, int[] inst) {
		for(ActivationSeq as: ass.a) {
			ActivationSet a=as.get(t);
			float r = 0;
			for(int i=0; i<in.count; i++){
				double trg = a.get(target, i);
				if (trg<=0) continue;
				double pp = a.get(in, i);
				r += trg*(Math.log(trg)-pp);
			}
			a.add(loss, 0, r);
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst!=null)
			throw new IllegalThreadStateException("A "+getClass().getName()+" module must not be inside a convolution");
		for(int b=0; b<as.length; ++b) {
			ActivationSet a=as.a[b].get(t);
			ActivationSet es = errors.a[b].get(t);
			float e=es.get(loss, 0);
			for(int i=0; i<in.count; i++){
				es.add(in, i, -a.get(target, i)*e);
			}			
		}
	}


//
////	@Override
//	public void declareDependencies(Dependencies d) {
//		d.declare(new InputDependency(in, this, 0));
//		d.declare(new InputDependency(target, this, 0));
//		d.declare(new OutputDependency(this, loss));
//		for(String phase: phases)
//			d.declareLoss(loss, phase);
//	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}
//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}


	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
}
