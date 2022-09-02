package org.siquod.ml.neural1.modules;

import java.util.Collections;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;

public class FormatCast extends InOutCastLayer{

	public FormatCast(Interface in2, int shift, TensorFormat outTf) {
		super(in2);
		if(in.count<shift + outTf.count())
			throw new IllegalArgumentException();
		out=new Interface(in.count, outTf);
		out.offset=in.offset+shift;
	}
	public static InOutCastFactory factory(final TensorFormat outTf){
		return factory(0, outTf);
	}
	public static InOutCastFactory factory(int shift, final TensorFormat outTf){
		return (in) -> new FormatCast(in, shift, outTf);
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		super.allocate(ia);
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
