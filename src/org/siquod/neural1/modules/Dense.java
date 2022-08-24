package org.siquod.neural1.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ActivationSeq;
import org.siquod.neural1.ActivationSet;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlock;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;
import org.siquod.neural1.modules.regularizer.Regularizer;

/**
 * This module performs a linear mapping from its input vecotr to its output vector
 * @author bb
 *
 */
public class Dense implements InOutBiasModule{

	Interface in;
	Interface out;
	ParamBlock weights;
	ParamBlock bias;
	ParamBlock[] coordinateBias;
	int[] shift;
	int[] posi;
	float[] coordinateActivations;
	boolean useBias;
	boolean[] useCoordinateBias;
	float biasInit=0;
	//	boolean useResidual;
	int dt;
	Regularizer reg;

	int par = 1;

	public Dense(){
		this(true, 0, null);
	}
	public Dense(boolean useBias){
		this(useBias, 0, null);
	}
	public Dense(int dt, int [] shift, boolean... useCoordinateBias){
		this(true, dt, shift, useCoordinateBias);
	}
	public Dense(boolean useBias, int dt, int [] shift, boolean... useCoordinateBias){
		this.useBias=useBias;
		this.dt=dt;
		this.shift=shift==null?null:shift.clone();
		posi=shift==null?null:new int[shift.length];
		this.useCoordinateBias=new boolean[useCoordinateBias.length];
		System.arraycopy(useCoordinateBias, 0, this.useCoordinateBias, 0, useCoordinateBias.length);
		this.coordinateBias=new ParamBlock[useCoordinateBias.length];

	}
	public Dense biasInit(float d) {biasInit=d; return this;}
	public void setCoordinateBiases(float[] cb) {
		coordinateActivations=cb;
	}
	public Dense regularizer(Regularizer r){
		reg=r;
		return this;
	}
	//	public FullyConnected useResidual(){
	//		useResidual=true;
	//		return this;
	//	}
	//	public FullyConnected useResidual(boolean b){
	//		useResidual=b;
	//		return this;
	//	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		//		if(useResidual && in.count!=out.count){
		//			System.err.println("Canot make this a residual layer: input and output have different sizes");
		//			useResidual=false;
		//		}
	}

	@Override
	public void allocate(ParamAllocator ia) {
		int incount = shift==null?in.count:in.channels();
		int outcount = shift==null?out.count:out.channels();
		int size = incount * outcount;
		weights = ia.allocate(new ParamBlock("weights", size));
		if(useBias)
			bias = ia.allocate(new ParamBlock("bias", outcount));
		for(int i=0; i<useCoordinateBias.length; ++i) {
			if(useCoordinateBias[i])
				coordinateBias[i]=ia.allocate(new ParamBlock("coordinateBias"+i, outcount));
		}
	}

	@Override
	public void share(ParamBlocks ps) {
		biasInit=Float.NaN;
		int size = in.channels() * out.channels();
		weights = ps.get("weights", size);
		if(useBias){
			bias = ps.get("bias", out.channels());
		}
		for(int i=0; i<useCoordinateBias.length; ++i) {
			if(useCoordinateBias[i])
				coordinateBias[i]=ps.get("coordinateBias"+i, out.channels());
		}
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("FullyConnected");
		ret.add(weights);
		if(useBias)
			ret.add(bias);
		for(int i=0; i<useCoordinateBias.length; ++i) {
			if(useCoordinateBias[i])
				ret.add(coordinateBias[i]);
		}
		return ret;
	}

	public static final ExecutorService parallelizer = Executors.newCachedThreadPool(); 
	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		int incount;
		int outcount;
		if(inst==null) {
			if(shift!=null)
				throw new IllegalArgumentException("This "+getClass().getName()+" module must be inside a convolution");
			incount = in.count;
			outcount = out.count;
		}else {
			if(shift==null)
				Module.copy(inst, posi);
			else
				Module.add(inst, shift, posi);
			incount = in.channels();
			outcount = out.channels();
		}
		if(par==1) {
			int startA = 0;
			int endA = as.a.length;
			int startO = 0;
			int endO = outcount;
			if(inst==null) {
				forwardSlice(params, as, t, incount, outcount, startA, endA, startO, endO);		
			}else {
				int[] poso=inst;
				forwardSlice(params, as, t, poso, incount, outcount, startA, endA, startO, endO);
			}
		}else {
			int taskCount = Math.min(par*4, as.a.length);
			AtomicInteger done = new AtomicInteger();
			ArrayList<Future<?>> workers = new ArrayList<>(par);
			try {
				for(int i=0; i<par; ++i) {
					workers.add(parallelizer.submit(()->{
						while(true) {
							int task = done.getAndAdd(1);
							if(task>=taskCount)
								return;
							int startA = task * as.a.length / taskCount;
							int endA = (task+1) * as.a.length / taskCount;
							int startO = 0;
							int endO = outcount;
							if(inst==null) {
								forwardSlice(params, as, t, incount, outcount, startA, endA, startO, endO);		
							}else {
								int[] poso=inst;
								forwardSlice(params, as, t, poso, incount, outcount, startA, endA, startO, endO);
							}
						}
					}));
				}
				joinAll(workers);

			}finally {
				for(Future<?> f: workers)
					f.cancel(true);
			}
		}
	}
	private void joinAll(ArrayList<Future<?>> workers) {
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
	private void forwardSlice(ParamSet params, ActivationBatch as, int t, int incount, int outcount, int startA,
			int endA, int startO, int endO) {
		for(int ai = startA; ai<endA; ++ai) {
			ActivationSeq a = as.a[ai];
			if(a==null) continue;
			ActivationSet oa = a.get(t);
			ActivationSet ia = a.get(t+dt);
			if(ia!=null){
				for(int i=incount-1; i>=0; --i){
					float input = ia.get(in, i);
					if(input==0) {
						continue;
					}
					for(int o=endO-1; o>=startO; --o){
						int w = o + i * outcount;
						oa.add(out, o, input * params.get(weights, w));
					}
				}
			}
			//		if(useResidual){
			//			for(int o=out.count-1; o>=0; --o){
			//				oa.add(out, o, ia.get(in, o));
			//			}
			//		}
			if(useBias){
				for(int o=endO-1; o>=startO; --o){
					oa.add(out, o, params.get(bias, o));
				}
			}
			for(int i=0; i<coordinateBias.length; ++i) {
				ParamBlock bias=coordinateBias[i];
				if(bias!=null) {
					float activation = coordinateActivations[i];
					for(int o=endO-1; o>=startO; --o){
						oa.add(out, o, activation*params.get(bias, o));
					}
				}
			}
		}
	}
	private void forwardSlice(ParamSet params, ActivationBatch as, int t, int[] poso, int incount, int outcount,
			int startA, int endA, int startO, int endO) {
		for(int ai = startA; ai<endA; ++ai) {
			ActivationSeq a = as.a[ai];
			if(a==null) continue;
			ActivationSet oa = a.get(t);
			ActivationSet ia = a.get(t+dt);
			if(ia!=null){
				for(int i=incount-1; i>=0; --i){
					float input = ia.get(in, posi, i);
					if(input==0) {
						continue;
					}
					for(int o=endO-1; o>=startO; --o){
						int w = o + i * outcount;
						oa.add(out, poso, o, input * params.get(weights, w));
					}
				}
			}
			//		if(useResidual){
			//			for(int o=out.count-1; o>=0; --o){
			//				oa.add(out, o, ia.get(in, o));
			//			}
			//		}
			if(useBias){
				for(int o=endO-1; o>=startO; --o){
					oa.add(out, poso, o, params.get(bias, o));
				}
			}
			for(int i=0; i<coordinateBias.length; ++i) {
				ParamBlock bias=coordinateBias[i];
				if(bias!=null) {
					float activation = coordinateActivations[i];
					for(int o=endO-1; o>=startO; --o){
						oa.add(out, poso, o, activation*params.get(bias, o));
					}
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {	
		if(dontBackprop.contains(phase))
			return;
		int incount;
		int outcount;
		if(inst==null) {
			incount = in.count;
			outcount = out.count;
		}else {
			if(shift==null)
				;
			else
				Module.add(inst, shift, posi);

			incount = in.channels();
			outcount = out.channels();
		}
		if(par==1) {
			int startA = 0;
			int endA = as.a.length;
			int startI = 0;
			int endI = incount;
			if(inst==null) {
				packPropSlice(params, errors, t, incount, outcount, startA, endA, startI, endI);
			}else {
				int[] poso=inst;
				backPropSlice(params, errors, t, incount, outcount, poso, startA, endA, startI, endI);
			}
		}else {
			int taskCount = Math.min(par*4, as.a.length);
			AtomicInteger done = new AtomicInteger();
			ArrayList<Future<?>> workers = new ArrayList<>(par);
			try {
				for(int i=0; i<par; ++i) {
					workers.add(parallelizer.submit(()->{
						while(true) {
							int task = done.getAndAdd(1);
							if(task>=taskCount)
								return;
							int startA = task * as.a.length / taskCount;
							int endA = (task+1) * as.a.length / taskCount;
							int startI = 0;
							int endI = incount;
							if(inst==null) {
								packPropSlice(params, errors, t, incount, outcount, startA, endA, startI, endI);
							}else {
								int[] poso=inst;
								backPropSlice(params, errors, t, incount, outcount, poso, startA, endA, startI, endI);
							}
						}
					}));
				}
				joinAll(workers);

			}finally {
				for(Future<?> f: workers)
					f.cancel(true);
			}
		}
	}
	private void backPropSlice(ParamSet params, ActivationBatch errors, int t, int incount, int outcount, int[] poso,
			int startA, int endA, int startI, int endI) {
		for(int b=startA; b<endA; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet inErr = errors.a[b].get(t+dt);
			if(inErr==null)
				continue;
			ActivationSet outErr = errors.a[b].get(t);
			if(outErr==null)
				continue;
			for(int i=endI-1; i>=startI; --i){
				float ie = 0;
				for(int o=outcount-1; o>=0; --o){
					int w = o + i * outcount;
					ie += outErr.get(out, poso, o)*params.get(weights, w);
				}
				//			if(useResidual){
				//				ie += outErr.get(out, i);
				//			}
				inErr.add(in, posi, i, ie);
			}
		}
	}
	private void packPropSlice(ParamSet params, ActivationBatch errors, int t, int incount, int outcount, int startA,
			int endA, int startI, int endI) {
		for(int b=startA; b<endA; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet inErr = errors.a[b].get(t+dt);
			if(inErr==null)
				continue;
			ActivationSet outErr = errors.a[b].get(t);
			if(outErr==null)
				continue;
			for(int i=endI-1; i>=startI; --i){
				float ie = 0;
				for(int o=outcount-1; o>=0; --o){
					int w = o + i * outcount;
					ie += outErr.get(out, o)*params.get(weights, w);
				}
				//			if(useResidual){
				//				ie += outErr.get(out, i);
				//			}
				inErr.add(in, i, ie);
			}
		}
	}



	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, 
			ActivationBatch errors, ParamSet gradients, int t, int[] inst) {
		int incount;
		int outcount;
		if(inst==null) {
			incount = in.count;
			outcount = out.count;
		}else {
			if(inst!=null)
				Module.add(inst, shift, posi);

			incount = in.channels();
			outcount = out.channels();
		}
		if(par==1) {
			int startI = 0;
			int endI = incount;
			int startO = 0;
			int endO = outcount;
			if(inst==null) {
				gradientsSlice(phase, as, errors, gradients, t, outcount, startI, endI, startO, endO);
			}else{
				int[] poso=inst;
				gradientsSlice(phase, as, errors, gradients, t, outcount, startI, endI, startO, endO, poso);
			}
		}else{
			int taskCount = Math.min(par*4, Math.max(incount, outcount));
			AtomicInteger done = new AtomicInteger();
			ArrayList<Future<?>> workers = new ArrayList<>(par);
			try {
				for(int i=0; i<par; ++i) {
					workers.add(parallelizer.submit(()->{
						while(true) {
							int task = done.getAndAdd(1);
							if(task>=taskCount)
								return;
							int startI, endI, startO, endO;
							
							if(incount>outcount){
								startI = task * incount / taskCount;
								endI = (task+1) * incount / taskCount;
								startO = 0;
								endO = outcount;
							}else {
								startO = task * outcount / taskCount;
								endO = (task+1) * outcount / taskCount;
								startI = 0;
								endI = incount;
							}
							
							if(inst==null) {
								gradientsSlice(phase, as, errors, gradients, t, outcount, startI, endI, startO, endO);
							}else{
								int[] poso=inst;
								gradientsSlice(phase, as, errors, gradients, t, outcount, startI, endI, startO, endO, poso);
							}
						}
					}));
				}
				joinAll(workers);

			}finally {
				for(Future<?> f: workers)
					f.cancel(true);
			}
		}
	}
	private void gradientsSlice(String phase, ActivationBatch as, ActivationBatch errors, ParamSet gradients, int t,
			int outcount, int startI, int endI, int startO, int endO, int[] poso) {
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet input = as.a[b].get(t+dt);
			ActivationSet outErr = errors.a[b].get(t);
			if(input!=null){
				if(weights.shouldLearn(phase)){
					for(int i=endI-1; i>=startI; --i){
						for(int o=endO-1; o>=startO; --o){
							int w = o + i * outcount;
							gradients.add(weights, w, outErr.get(out, poso, o) * input.get(in, posi, i));
						}
					}
				}
			}
			if(startI==0) {

				if(useBias){
					if(bias.shouldLearn(phase)){
						for(int o=endO-1; o>=startO; --o){
							gradients.add(bias, o, outErr.get(out, poso, o));
						}
					}			
				}
				for(int i=0; i<coordinateBias.length; ++i) {
					ParamBlock bias=coordinateBias[i];
					if(bias!=null) {
						float activation = coordinateActivations[i];
						for(int o=endO-1; o>=startO; --o){
							gradients.add(bias, o, activation*outErr.get(out, poso, o));
						}
					}
				}
			}
		}
	}
	private void gradientsSlice(String phase, ActivationBatch as, ActivationBatch errors, ParamSet gradients, int t,
			int outcount, int startI, int endI, int startO, int endO) {
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet input = as.a[b].get(t+dt);
			ActivationSet outErr = errors.a[b].get(t);
			if(input!=null){
				if(weights.shouldLearn(phase)){
					for(int i=endI-1; i>=startI; --i){
						for(int o=endO-1; o>=startO; --o){
							int w = o + i * outcount;
							gradients.add(weights, w, outErr.get(out, o) * input.get(in, i));
						}
					}
				}
			}
			if(startI==0) {
				if(useBias){
					if(bias.shouldLearn(phase)){
						for(int o=endO-1; o>=startO; --o){
							gradients.add(bias, o, outErr.get(out, o));
						}
					}			
				}
				for(int i=0; i<coordinateBias.length; ++i) {
					ParamBlock bias=coordinateBias[i];
					if(bias!=null) {
						float activation = coordinateActivations[i];
						for(int o=endO-1; o>=startO; --o){
							gradients.add(bias, o, activation*outErr.get(out, o));
						}
					}
				}
			}
		}
	}
	@Override
	public void regularize(String phase, ParamSet params, ParamSet gradients, float globReg) {
		if(reg!=null && weights.shouldLearn(phase)){
			reg.regularize(params, gradients, weights.start, out.count, 1, in.count, out.count, globReg);
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
	//	@Override
	public boolean wouldBackprop(String phase) {
		return !dontBackprop.contains(phase);
	}
	HashSet<String> dontBackprop=new HashSet<>();

	public Dense dontBackprop(String phase){
		dontBackprop.add(phase);
		return this;
	}
	public String showParams(ParamSet ps) {
		StringBuilder ret=new StringBuilder();
		ret.append("matrix:\n");
		for(int i=0, w=0; i<in.count; ++i){
			for(int o=0; o<out.count; ++o, ++w){
				ret.append(ps.get(weights, w)).append(", \t");
			}
			ret.append("\n");
		}
		ret.append("bias:\n");
		if(useBias)
			for(int o=0; o<out.count; ++o){
				ret.append(ps.get(bias, o)).append(", \t");
			}

		return ret.toString();
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
	@Override
	public void initParams(ParamSet p) {
		if(bias!=null && !Double.isNaN(biasInit))
			for(int i=0; i<bias.count; ++i)
				p.set(bias, i, biasInit);
	}
	public ParamBlock getBias() {
		return bias;
	}
	public Dense dt(int d){
		dt=d;
		return this;
	}
	public Dense shift(int[] shift){
		this.shift=shift==null?null:shift.clone();
		posi=shift==null?null:new int[shift.length];
		return this;
	}
	public Dense parallel(int threads) {
		if(threads<1)
			throw new IllegalArgumentException();
		par = threads;
		return this;
	}
} 
