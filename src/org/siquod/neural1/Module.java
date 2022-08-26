package org.siquod.neural1;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

public interface Module {
	public static final ExecutorService parallelizer = Executors.newCachedThreadPool(); 

	public static int[] EIA= {};
	void allocate(InterfaceAllocator ia);
	void allocate(ParamAllocator ia);
	void share(ParamBlocks ps);
	ParamBlocks getParamBlocks();
	void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst);
	void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst);
//	public void declareDependencies(Dependencies d);
	public void dontComputeInPhase(String phase);
//	public boolean wouldBackprop(String phase);

	static void add(int[] in1, int[] in2, int[] out) {
		for(int i=0; i<in1.length; ++i)
			out[i]=in1[i]+in2[i];
	}
	static void copy(int[] in, int[] out) {
		System.arraycopy(in, 0, out, 0, in.length);
	}
	default void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients, int t, int[] inst) {
		for(Module m: getSubmodules())
			m.gradients(phase, params, as, errors, gradients, t, inst);
	}
	default void regularize(String phase, ParamSet params, ParamSet gradients, float globReg) {
		for(Module m: getSubmodules())
			m.regularize(phase, params, gradients, globReg);
	}
	default void allocateStatistics(InterfaceAllocator ia) {
		for(Module m: getSubmodules()) {
			m.allocateStatistics(ia);
		}
	}
	default void updateStatistics(ActivationSeq stat, ParamSet params, Function<Integer, Float> owt, float[] weight, int tMin) {
		for(Module m: getSubmodules())
			m.updateStatistics(stat, params, owt, weight, tMin);
	}
	//	default void initializeBatch(ActivationSeq as) {
//		for(Module m: getSubmodules())
//			m.initializeBatch(as);
//	}
	default void initializeRun(ActivationBatch as, boolean training) {
		for(Module m: getSubmodules())
			m.initializeRun(as, training);
	}
	public abstract List<Module> getSubmodules();
	public default List<Module> deepSubmodules() {
		List<Module> ret = new ArrayList<>();
		giveDeepSubmodules(ret);
		return ret;
	}
	default void giveDeepSubmodules(List<Module> ret) {
		ret.add(this);
		for(Module m: getSubmodules()) {
			m.giveDeepSubmodules(ret);
		}
	}
	public default void initParams(ParamSet p) {
		defaultInitParams(p);
	}
	public default void defaultInitParams(ParamSet p) {
		for(Module m: getSubmodules())
			m.initParams(p);
	}
	public static void joinAll(ArrayList<Future<?>> workers) {
		try {
			for(Future<?> f: workers) 
				f.get();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		} catch (ExecutionException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}
}
