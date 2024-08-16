package org.siquod.ml.neural1.modules;

import java.util.List;
import java.util.function.Predicate;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;

public class Switch implements InOutModule{

	public static final Predicate<String> NEVER = s -> false;
	public static final Predicate<String> ALWAYS= s -> false;
	Predicate<String> gradientPhases=NEVER;
	Predicate<String> backpropPhases=ALWAYS;
	Predicate<String> forwardPhases=ALWAYS;
	InOutModule inner;
	Interface in, out;
	
	public Switch(Switch copyThis) {
		gradientPhases=copyThis.gradientPhases;
        backpropPhases=copyThis.backpropPhases;
        forwardPhases=copyThis.forwardPhases;
        inner=copyThis.inner.copy();
        out=copyThis.out;
        in=copyThis.in;
	}
	@Override
	public InOutModule copy() {
		return new Switch(this);
	}
	
	public Switch() {
		
	}
	
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		inner.allocate(ia);
	}

	@Override
	public void allocate(ParamAllocator ia) {
		inner.allocate(ia);
	}

	@Override
	public void share(ParamBlocks ps) {
		inner.share(ps);
	}

	@Override
	public ParamBlocks getParamBlocks() {
		return inner.getParamBlocks();
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void dontComputeInPhase(String phase) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public List<Module> getSubmodules() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Interface getIn() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Interface getOut() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int dt() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int[] shift() {
		// TODO Auto-generated method stub
		return null;
	}

}
