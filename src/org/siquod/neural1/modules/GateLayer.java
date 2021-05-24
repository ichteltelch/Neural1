package org.siquod.neural1.modules;

import java.util.Collections;
import java.util.HashMap;
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
 * This Module performs point-wise multiplication of its two inputs
 * @author bb
 *
 */
public class GateLayer implements Module{

	Interface in1, in2, out;
	int dt1, dt2;
	int[] shift1;
	int[] posi1;
	int[] shift2;
	int[] posi2;


	public GateLayer(){
		this(0, 0, null, null);
	}
	public GateLayer(int dt1, int dt2){
		this(dt1, dt2, null, null);
	}
	public GateLayer(int dt1, int dt2, int[] s1, int[] s2){
		this.dt1=dt1;
		this.dt2=dt2;
		this.shift1=shift1==null?null:shift1.clone();
		posi1=shift1==null?null:new int[shift1.length];
		this.shift2=shift2==null?null:shift2.clone();
		posi2=shift2==null?null:new int[shift2.length];


	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in1=ia.get("in1");
		in2=ia.get("in2", in1.count);
		out=ia.get("out", in1.count);
		if(out.tf==null)
			out.tf=in1.tf;
		if(out.tf==null)
			out.tf=in2.tf;
	}
	public void allocate(InterfaceAllocator ia, String in1, String in2, String out){
		HashMap<String, String> m = new HashMap<>(3);
		m.put("in1", in1);
		m.put("in2", in2);
		m.put("out", out);
		ia.push(m);
		allocate(ia);
		ia.pop();
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
			int incount = in1.count;
			for(ActivationSeq a: as.a) {
				if(a==null) continue;

				ActivationSet a1=a.get(t+dt1);
				ActivationSet a2=a.get(t+dt2);
				ActivationSet ao=a.get(t);
				if(a1==null || a2==null)
					continue;

				// o = i1 * i2
				for(int i=0; i<incount; ++i) {
					ao.add(out, i, a1.get(in1, i) * a2.get(in2, i));
				}
			}
		}else {
			if(shift1==null) {
				Module.copy(inst, posi1);
			}else {
				Module.add(inst, shift1, posi1);
			}
			if(shift2==null) {
				Module.copy(inst, posi2);
			}else {
				Module.add(inst, shift2, posi2);
			}

			int[] poso=inst;
			int incount = in1.channels();
			for(ActivationSeq a: as.a) {
				if(a==null) continue;

				ActivationSet a1=a.get(t+dt1);
				ActivationSet a2=a.get(t+dt2);
				ActivationSet ao=a.get(t);
				if(a1==null || a2==null)
					continue;

				// o = i1 * i2
				for(int i=0; i<incount; ++i) {
					ao.add(out, poso, i, a1.get(in1, posi1, i) * a2.get(in2, posi2, i));
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst==null) {
			int incount = in1.count;
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a1=as.a[b].get(t+dt1);
				ActivationSet a2=as.a[b].get(t+dt2);
				ActivationSet e1=errors.a[b].get(t+dt1);
				ActivationSet e2=errors.a[b].get(t+dt2);
				ActivationSet eo=errors.a[b].get(t);
				// d l / d i1 = (d l / d o) * (d o / d i1) = (d l / d o) * i2 
				if(e1!=null && eo!=null && a2!=null){
					for(int i=0; i<incount; ++i) {
						e1.add(in1, i, eo.get(out, i) * a2.get(in2, i));
					}
				}
				if(e2!=null && eo!=null && a1!=null){
					for(int i=0; i<incount; ++i) {
						e2.add(in2, i, eo.get(out, i) * a1.get(in1, i));
					}
				}
			}
		}else {
			if(shift1==null) {
				Module.copy(inst, posi1);
				Module.copy(inst, posi2);
			}else {
				Module.add(inst, shift1, posi1);
				Module.add(inst, shift2, posi2);
			}
			int[] poso=inst;
			int incount = in1.channels();
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a1=as.a[b].get(t+dt1);
				ActivationSet a2=as.a[b].get(t+dt2);
				ActivationSet e1=errors.a[b].get(t+dt1);
				ActivationSet e2=errors.a[b].get(t+dt2);
				ActivationSet eo=errors.a[b].get(t);
				// d l / d i1 = (d l / d o) * (d o / d i1) = (d l / d o) * i2 
				if(e1!=null && eo!=null && a2!=null){
					for(int i=0; i<incount; ++i) {
						e1.add(in1, posi1, i, eo.get(out,poso, i) * a2.get(in2, posi2, i));
					}
				}
				if(e2!=null && eo!=null && a1!=null){
					for(int i=0; i<incount; ++i) {
						e2.add(in2, posi2, i, eo.get(out, poso, i) * a1.get(in1, posi1, i));
					}
				}
			}
		}
	}


//	//	@Override
//	public void declareDependencies(Dependencies d) {
//		d.declare(new InputDependency(in1, this, dt1));
//		d.declare(new InputDependency(in2, this, dt2));
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
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
}
