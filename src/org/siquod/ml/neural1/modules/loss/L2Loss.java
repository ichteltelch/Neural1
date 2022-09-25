package org.siquod.ml.neural1.modules.loss;

import java.util.Arrays;
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

public class L2Loss extends LossLayer{
	Interface in, target, loss;
	List<String> phases;
	LossGroup[] lgs;
	TensorFormat tf;

	public L2Loss(String... ph) {
		this(null, ph);
	}


	public L2Loss(LossGroup[] lgs, String... ph) {
		this.lgs=lgs;
	}


	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		target=ia.get("target", in.count);
		loss=ia.get("loss");
		tf=in.tf;
		while(tf.rank>2) {
			tf = tf.flattenIndexAndNext(1);
		}
		if(tf.rank==1) {
			tf = tf.insertUnitIndex(0);
		}
		int channels = tf.channels();
		if(lgs==null)
			lgs=LossGroup.makeDefault(channels);
		else {
			int lgl = lgs[lgs.length-1].end;
			if(lgl > channels) {
				throw new IllegalStateException("There are not enough channels for the specified LossGroups");
			}else if(lgl < channels) {
				System.err.println("Warning: There are more channels than specified by the LossGroups");
			}
		}
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
		int n0 = tf.dims[0];
		for(ActivationSeq as: ass) {
			ActivationSet a=as.get(t);
			float r = 0;
			for(int bri = 0; bri<n0; ++bri) {
				for(LossGroup lg: lgs) {
					if(lg.weight==0)
						continue;
					float gate;
					if(lg.isGated()) {
						gate =  a.get(target, tf.index(bri, lg.gate));
						if(lg.gateInverted)
							gate = 1-gate;
						if(gate==0)
							continue;
						gate *= lg.weight;
					}else {
						gate = lg.weight;
					}
					float rr = 0;
					{
						for(int i=lg.start; i<lg.end; i++){
							int index = tf.index(bri, i);
							double trg = a.get(target, index);
							double pp = a.get(in, index);
							double diff = trg - pp;
							rr += diff*diff;


						}
					}
					r += rr*gate;
				}
			}
			a.add(loss, 0, r);

		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst!=null)
			throw new IllegalThreadStateException("A "+getClass().getName()+" module must not be inside a convolution");
		int n0 = tf.dims[0];
		for(int b=0; b<as.length; ++b) {
			ActivationSet a=as.a[b].get(t);
			ActivationSet es = errors.a[b].get(t);
			float e=es.get(loss, 0);
			for(int bri = 0; bri<n0; ++bri) {
				for(LossGroup lg: lgs) {
					if(lg.weight==0)
						continue;
					float gate;
					if(lg.isGated()) {
						gate =  a.get(target, tf.index(bri, lg.gate));
						if(lg.gateInverted)
							gate = 1-gate;
						if(gate==0)
							continue;
						gate *= lg.weight;
					}else {
						gate = lg.weight;
					}
					float ge = gate*e;
					
					{
						for(int i=lg.start; i<lg.end; i++){
							int index = tf.index(bri, i);
							double trg = a.get(target, index);
							if (trg<=0) continue;
							double pp = a.get(in, index);
							double diff = pp-trg;
							
							es.add(in, index, (float)(ge *   2 * diff));


						}

					}
					

				}
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
