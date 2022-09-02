package org.siquod.ml.neural1.modules.loss;


import java.util.Arrays;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.modules.LogSoftmax;

/**
 * This loss module combines a softmax layer with a nnl loss layer
 * @author bb
 *
 */
public class SoftMaxNllLoss extends LossLayer{
	public static final String hiddenName="LogSoftMax output";
	LogSoftmax sm;
	NllLoss loss;
	Interface hidden;
	public SoftMaxNllLoss(String... ph){
		this(null, ph);
	}
	public SoftMaxNllLoss(LossGroup[] lgs, String... ph){
		sm=new LogSoftmax(lgs);
		loss=new NllLoss(lgs, ph);
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		Interface in=ia.get("in");
		hidden=ia.allocate(new Interface(hiddenName, in.count, null));
		sm.allocate(ia, "in", hidden.name);
		loss.allocate(ia, hidden.name, "target", "loss");
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
		sm.forward(training, params, as, t, inst);
		loss.forward(training, params, as, t, inst);
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		loss.backprop(phase, params, as, errors, t, inst);
		sm.backprop(phase, params, as, errors, t, inst);
	}

////	@Override
//	public void declareDependencies(Dependencies d) {
//		sm.declareDependencies(d);
//		loss.declareDependencies(d);
//	}
	@Override
	public void dontComputeInPhase(String phase) {
		hidden.dontComputeInPhase(phase);
	}
//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}
	@Override
	public List<Module> getSubmodules() {
		return Arrays.asList(sm, loss);
	}
}
