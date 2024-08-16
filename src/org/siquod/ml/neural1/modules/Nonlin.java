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
import org.siquod.ml.neural1.neurons.Neuron;

/**
 * This module applies the same nonlinearity to all its input elements
 * @author bb
 *
 */
public class Nonlin implements InOutModule{
	Neuron n;
	Interface in;
	Interface out;

	public Nonlin(Neuron n){
		this.n=n;
	}
	public Nonlin(Nonlin copyThis) {
		this.n=copyThis.n;
		this.in=copyThis.in;
		this.out=copyThis.out;
	}
	public Nonlin copy() {
		return this;
	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out", in.count);
		if(out.tf==null)
			out.tf=in.tf;

	}

	@Override
	public void allocate(ParamAllocator ia) {
		return;
	}

	@Override
	public void share(ParamBlocks ps) {
		return;
	}

	@Override
	public ParamBlocks getParamBlocks() {
		return null;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] pos) {
		if(pos==null) {
			int incount = in.count;
			for(ActivationSeq b: as) {
				if(b==null) continue;

				ActivationSet a=b.get(t);
				for(int i=0; i<incount; ++i)
					a.add(out, i, n.f(a.get(in, i))); 
			}
		}else {
			int incount = in.channels();
			for(ActivationSeq b: as) {
				if(b==null) continue;

				ActivationSet a=b.get(t);
				for(int i=0; i<incount; ++i)
					a.add(out, pos, i, n.f(a.get(in, pos, i))); 
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] pos) {
		if(pos==null) {
			int incount = in.count;
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int i=0; i<incount; ++i)
					e.add(in, i, e.get(out, i) * n.df(a.get(in, i))); 
			}
		}else {
			int incount = in.channels();
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int i=0; i<incount; ++i)
					e.add(in, pos, i, e.get(out, pos, i) * n.df(a.get(in, pos, i))); 
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
