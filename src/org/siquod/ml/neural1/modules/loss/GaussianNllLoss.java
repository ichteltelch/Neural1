package org.siquod.ml.neural1.modules.loss;

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

public class GaussianNllLoss extends LossLayer{
	static final double schwitz=1;
	private static final double lnSqrtTau = 0.5*Math.log(2*Math.PI);
	Interface in, target, loss;
	@Override
	public LossLayer copy() {
		return this;
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		int ic = in.count;
		if((ic&1)!=0)
			throw new IllegalArgumentException("GaussianNllLoss layers need an even number of inputs!");
		target=ia.get("target", ic/2);
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
		for(ActivationSeq as: ass) {
			ActivationSet a=as.get(t);
			float r = 0;
			for(int i=0, ti=0; i<in.count; i+=2, ++ti){
				double µ = a.get(in, i);
				double q = a.get(in, i+1)*schwitz;


				double trg = a.get(target, ti);

				double diff = µ-trg;
				double invSigmaSquare = Math.exp(-2*q);
				// - ln (exp(-diff²/sigma²)/(sqrt(tau)*sigma))
				double nll = 0.5*diff*diff*invSigmaSquare + q + lnSqrtTau;
				if(false && i==10 && training==ForwardPhase.TESTING && 
						++counter > 100) {
//					System.out.println("("+µ+", "+q+") <-> "+trg+": "+nll);
					System.out.println(Math.signum(µ));
					counter=0;
				}
				r += nll; 
			}
			a.add(loss, 0, r*2f/in.count);
		}
	}
	int counter = 0;

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst!=null)
			throw new IllegalThreadStateException("A "+getClass().getName()+" module must not be inside a convolution");
		counter=0;
		for(int b=0; b<as.length; ++b) {
			ActivationSet a=as.a[b].get(t);
			ActivationSet es = errors.a[b].get(t);
			float e=es.get(loss, 0)*2f/in.count;
			for(int i=0, ti=0; i<in.count; i+=2, ++ti){
				double µ = a.get(in, i);
				double q = a.get(in, i+1)*schwitz;


				double trg = a.get(target, ti);

				double diff = µ-trg;
				double invSigmaSquare = Math.exp(-2*q);
				// - ln (exp(-diff²/sigma²)/(sqrt(tau)*sigma))
				double val = 0.5*diff*diff*invSigmaSquare + q + lnSqrtTau;
				double dValDµ = diff*invSigmaSquare;
				double dValDq = -diff*diff*invSigmaSquare + 1;
				es.add(in, i, (float) (dValDµ*e));
				es.add(in, i+1, (float) (dValDq*e*schwitz));

				if(false && b==0 && i==0) {
					double check;
					{
						double sigma = Math.exp(q);
						check = - Math.log((1/(Math.sqrt(2*Math.PI)*sigma)) * Math.exp(-diff*diff/(2*sigma*sigma)) );
					}


					System.out.println("("+µ+", "+q+") <-> "+trg+": "+val+"(== "+check+"); => -("+dValDµ+", "+dValDq+")");
				}

			}			
		}
	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}
	public boolean wouldBackprop(String phase) {
		return true;
	}
	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}

}
