package org.siquod.neural1.modules;

import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;

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

public class BackpropStopper extends InOutCastLayer{

	public static final Predicate<String> NEVER = s -> false;
	public static final Predicate<String> ALWAYS= s -> false;
	Predicate<String> backprop=NEVER;
	private int dt;
	private int[] shift, posi;

	public BackpropStopper(Interface in, String outName, int dt, int[] shift){
		super(in);
		out=new Interface(outName, in.count, in.tf);
		out.offset=in.offset;	
		this.dt=dt;
		this.shift=shift==null?null:shift.clone();
		posi=shift==null?null:new int[shift.length];

	}
	public BackpropStopper(Interface in, String outName){
		this(in, outName, 0, null);
	}


	@Override
	public void allocate(InterfaceAllocator ia) {
		if(out.tf==null)
			out.tf=in.tf;
		out=ia.allocate(out);
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
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(!backprop.test(phase)){
			if(inst==null) {
				for(ActivationSeq a: errors)
					if(a!=null) {
						ActivationSet aa = a.get(t+dt);
						if(aa!=null)
							aa.clear(in);
					}
			}else {
				int[] pos;
				if(shift!=null)
					Module.add(inst, shift, pos=posi);
				else
					pos=inst;
				for(ActivationSeq a: errors)
					if(a!=null) {
						ActivationSet aa = a.get(t+dt);
						if(aa!=null)
							aa.clearAllChannels(in, pos);
					}						
			}
		}
	}




	@Override
	public void dontComputeInPhase(String phase) {		
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
