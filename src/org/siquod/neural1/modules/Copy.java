package org.siquod.neural1.modules;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ActivationSeq;
import org.siquod.neural1.ActivationSet;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.InputDependency;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.OutputDependency;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;

/**
 * This module simply copies its input, but possibly form a different instant in time.
 * @author bb
 *
 */
public class Copy implements InOutModule{
	int dt;
	int[] shift;
	int[] posi;
	Interface in;
	Interface out;
	HashSet<String> dontBackprop=new HashSet<>();
	public int width=-1;
	public Copy(int dt, int [] shift){
		this.dt=dt;
		this.shift=shift==null?null:shift.clone();
		posi=shift==null?null:new int[shift.length];
	}
	public Copy(){
		this(0, null);
	}
	public Copy dontBackprop(String phase){
		dontBackprop.add(phase);
		return this;
	}
	//	public void allocate(InterfaceAllocator ia, String in, String out){
	//		HashMap<String, String> m = new HashMap<>(2);
	//		m.put(in, "in");
	//		m.put(in, "out");
	//		ia.push(m);
	//		allocate(ia);
	//		ia.pop();
	//	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		int incount = shift==null?in.count:in.channels();
		int outcount = shift==null?out.count:out.channels();
		if(out.tf==null && incount==outcount)
			out.tf=in.tf;
		if(width==-1){
			width=Math.min(incount, outcount);
		}else{
			if(width>incount || width>outcount)
				throw new IllegalArgumentException("manually set width is too high");
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
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(inst==null) {
			if(shift!=null)
				throw new IllegalArgumentException("This "+getClass().getName()+" module must be inside a convolution");

			for(ActivationSeq a: as.a) {
				if(a==null) continue;
				ActivationSet ia=a.get(t+dt);
				if(ia==null)
					continue;
				ActivationSet oa=a.get(t);
				for(int i=0; i<width; ++i) {
					oa.add(out, i, ia.get(in, i));
				}
			}

		}else {
			if(shift==null)
				Module.copy(inst, posi);
			else
				Module.add(inst, shift, posi);
			int[] poso=inst;
			for(ActivationSeq a: as.a) {
				if(a==null) continue;
				ActivationSet ia=a.get(t+dt);
				if(ia==null)
					continue;
				ActivationSet oa=a.get(t);
				for(int i=0; i<width; ++i) {
					oa.add(out, poso, i, ia.get(in, posi, i));
				}

			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(dontBackprop.contains(phase))
			return;
		if(inst==null) {
			for(ActivationSeq e: errors.a) {
				if(e==null) continue;
				ActivationSet ie=e.get(t+dt);
				if(ie==null) continue;
				ActivationSet oe=e.get(t);
				for(int i=0; i<width; ++i) {
					ie.add(in, i, oe.get(out, i));
				}
			}
		}else {
			if(shift!=null)
				Module.add(inst, shift, posi);
			else
				Module.copy(inst, posi);
			int[] poso=inst;
			for(ActivationSeq e: errors.a) {
				if(e==null) continue;
				ActivationSet ie=e.get(t+dt);
				if(ie==null) continue;

				ActivationSet oe=e.get(t);
				for(int i=0; i<width; ++i) {
					in.add(ie, posi, i, out.get(oe, poso, i));
				}
			}
		}
	}
//	//	@Override
//	public void declareDependencies(Dependencies d) {
//		d.declare(new InputDependency(in, this, dt));
//		d.declare(new OutputDependency(this, out));
//	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}
	public boolean wouldBackprop(String phase) {
		return !dontBackprop.contains(phase);
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
		return dt;
	}
	@Override
	public int[] shift() {
		return shift;
	}
	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
}
