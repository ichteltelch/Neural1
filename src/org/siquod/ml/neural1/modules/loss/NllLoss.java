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

/**
 * This negative log-likelihood loss layer expects a logarithmized probability distribution in 
 * its input interface "in", a probability distribution in its input interface "target" and 
 * outputs the loss in its one-dimensional output interface "loss". 
 * 
 * It is possible to sum the losses from multiple distributions using {@link LossGroup}s 
 * @author bb
 *
 */
public class NllLoss extends LossLayer{
	Interface in, target, loss;
	List<String> phases;
	LossGroup[] lgs;
	TensorFormat tf;

	public NllLoss(String... ph) {
		this(null, ph);
	}


	public NllLoss(LossGroup[] lgs, String... ph) {
		phases=Arrays.asList(ph.clone());
		this.lgs=lgs;
	}


	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		target=ia.get("target", in.count);
		loss=ia.get("loss", 1);
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
					if(lg.isSingleton()) {
						int i = lg.start;
						int index = tf.index(bri, i);
						double trg = a.get(target, index);
						double pp = a.get(in, index);

						if (trg>0) {
							rr += trg*(Math.log(trg)-Math.log(pp));
						}
						if(trg<1) {
							rr += (1-trg)*(Math.log(1-trg)-Math.log(1-pp));
						}
						if(!Float.isFinite(rr)) {
							System.out.println();
							rr = 0;
						}
					}else {
						for(int i=lg.start; i<lg.end; i++){
							int index = tf.index(bri, i);
							double trg = a.get(target, index);
							if (trg<=0) continue;
							double pp = a.get(in, index);
							rr += trg*(Math.log(trg)-pp);
							if(!Float.isFinite(rr)) {
								System.out.println();
							}

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
					
					if(lg.isSingleton()) {
						int i = lg.start;
						int index = tf.index(bri, i);
						float trg = a.get(target, index);
						float pp = a.get(in, index);
						
						float ppe=0;
						if (trg>0) {
							ppe += trg*(-1/(pp));
						}
						if(trg<1) {
							ppe += (1-trg)*(1/(1-pp));
						}
						if(!Float.isFinite(ppe)) {
							System.out.println();
							ppe=0;
						}
						es.add(in, index, ppe*ge);

					}else {
//						for(int i=lg.start; i<lg.end; i++){
//							int index = tf.index(bri, i);
//							double trg = a.get(target, index);
//							if (trg<=0) continue;
//							double pp = a.get(in, index);
//							rr += trg*(Math.log(trg)-pp);
//							if(!Float.isFinite(rr)) {
//								System.out.println();
//							}
//
//						}
						for(int i=lg.start; i<lg.end; i++){
							int index = tf.index(bri, i);
							es.add(in, index, -a.get(target, index)*ge);
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
