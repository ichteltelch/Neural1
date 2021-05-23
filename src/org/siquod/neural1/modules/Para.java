package org.siquod.neural1.modules;

import java.util.Arrays;
import java.util.List;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;

public class Para implements InOutModule{

	InOutModule[] exec;
	private Interface in;
	private Interface out;
	public Para(InOutModule... sub) {
		exec=sub.clone();
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		for(InOutModule m: exec)
			m.allocate(ia, "in", "out");
		
	}

	@Override
	public void allocate(ParamAllocator ia) {
		for(InOutModule m: exec) {
			ia.push(null); m.allocate(ia); ia.pop();
		}
	}

	@Override
	public void share(ParamBlocks ps) {
		for(int i=0; i<exec.length; ++i)
			exec[i].share(ps.get(""+i));
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret=new ParamBlocks("LSTM");
		for(int i=0; i<exec.length; ++i)
			ret.add(""+i, exec[i].getParamBlocks());
		return ret;	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		for(int i=0; i<exec.length; ++i)
			exec[i].forward(training, params, as, t, inst);
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int i=exec.length-1; i>=0; --i)
			exec[i].backprop(phase, params, as, errors, t, inst);
	}

	@Override
	public void dontComputeInPhase(String phase) {
		for(InOutModule m: exec)
			m.dontComputeInPhase(phase);
	}

	@Override
	public List<Module> getSubmodules() {
		return Arrays.asList(exec);
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

}
