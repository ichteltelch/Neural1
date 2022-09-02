package org.siquod.ml.neural1.modules;

import java.util.Collections;
import java.util.List;
import java.util.Random;

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

public class Dropout extends InOutCastLayer{
	public float keepProbability;
	public int dropoutOffset=-1;
	Random r;
	public Dropout(Interface in2, double p, Random r) {
		super(in2);
		keepProbability=(float) p;
		out=new Interface(in.count, in.tf);
		out.offset=in.offset;
		this.r=new Random(r.nextLong());

	}
	public static InOutCastFactory factory(final double keepProb){
		return factory(keepProb, new Random());
	}
	public static InOutCastFactory factory(final double keepProb, Random rand){
		Random r=new Random(rand.nextLong());
		return new InOutCastFactory() {
			@Override
			public InOutCastLayer produce(Interface in) {
				return new Dropout(in, keepProb, r);
			}
		};
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		super.allocate(ia);
		if(dropoutOffset==-1)
			dropoutOffset=ia.allocateDropout(in.count);
		if(out.tf==null)
			out.tf=in.tf;

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
		if(inst!=null)
			throw new IllegalArgumentException("A "+getClass().getName()+" module must not be inside a convolution");
		if(training==ForwardPhase.TRAINING){
			for(ActivationSeq ab: as) {
				if(ab==null) continue;
				ActivationSet a=ab.get(t);
				for(int i=0; i<in.count; ++i){
					a.mult(in, i, ab.getDropout(dropoutOffset + i));
				}		

			}
		}else{
			for(ActivationSeq ab: as) {
				if(ab==null) continue;
				ActivationSet a=ab.get(t);
				for(int i=0; i<in.count; ++i){
					a.mult(in, i, keepProbability);
				}
			}
		}
	}
	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet e=errors.a[b].get(t);
			ActivationSeq asb = as.a[b];
			for(int i=0; i<in.count; ++i){
				e.mult(in, i, asb.getDropout(dropoutOffset + i));
			}		
		}
	}

	@Override
	public void dontComputeInPhase(String phase) {
	}
//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}


	@Override
	public void initializeRun(ActivationBatch as, boolean training) {
		if(training)
			for(ActivationSeq a: as)
				a.sampleDropout(dropoutOffset, in.count, keepProbability, r);
//				a.sampleDropout(dropoutOffset, in.count, keepProbability);
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
