package org.siquod.ml.neural1.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ActivationSeq;
import org.siquod.ml.neural1.ActivationSet;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlock;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;

public class BatchReNorm implements InOutScaleBiasModule{

	private static final float epsilon = 1e-10f;
	private Interface in;
	private Interface out;
	private Interface mean, sdev;
	ParamBlock add, mult, runningMean, runningSdev;
	int age=0;
	boolean hasAdd=true;
	boolean broadcast;
	int par=1;
	TensorFormat tf;

	Function<Integer, Float> rmaxSchedule=rmaxSchedule(100);
	Function<Integer, Float> dmaxSchedule=dmaxSchedule(100);

	static Function<Integer, Float> rmaxSchedule(int scal){
		return n -> n < scal ? 1 : n <8*scal ?  1+2*(n-scal)/7f:3;
	}	
	static Function<Integer, Float> dmaxSchedule(int scal){
		return n -> n < scal ? 0 : n <5*scal ?  5*(n-scal)/4f:5;
	}
	public BatchReNorm(boolean hasAdd, boolean broadcast) {
		this.hasAdd=hasAdd;
		this.broadcast=broadcast;
	}
	public BatchReNorm() {
		this(true, true);
	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		if(!in.tf.equals(out.tf))
			throw new IllegalArgumentException("input and output layer must be of the same size");
		if(broadcast) {
			tf=in.tf.to2D();		
		}else{
			tf = new TensorFormat(1, in.count);
		}
	}

	@Override
	public void allocateStatistics(InterfaceAllocator ia) {
		ia.push(null);
		int ch = tf.channels();
		mean = ia.allocate(new Interface("mean", ch, null));
		sdev = ia.allocate(new Interface("sdev", ch, null));
		ia.pop();
	}
	@Override
	public void allocate(ParamAllocator ia) {
		int ch = tf.channels();
		if(hasAdd) add = ia.allocate(new ParamBlock("add", ch));
		mult = ia.allocate(new ParamBlock("mult", ch));
		runningMean = ia.allocate(new ParamBlock("mean", ch));
		runningSdev = ia.allocate(new ParamBlock("sdev", ch));
	}

	@Override
	public void share(ParamBlocks ps) {
		int ch = tf.channels();
		if(hasAdd) add = ps.get("add", ch);
		mult = ps.get("mult", ch);		
		runningMean = ps.get("mean", ch);
		runningSdev = ps.get("sdev", ch);		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("BatchReNorm");
		if(hasAdd) ret.add(add);
		ret.add(mult);
		ret.add(runningMean);
		ret.add(runningSdev);
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		ActivationSet bparm = as.batchParams.get(t);
		if(inst==null) {
			int ch = tf.channels();
			if(training != ForwardPhase.TESTING) {

				float dmax=dmaxSchedule.apply(age);
				float rmax=rmaxSchedule.apply(age);

				int taskCount = Math.min(par*4, ch);
				if(par==1 || taskCount==1) {
					int startI = 0;
					int endI = ch;
					int startBri = 0;
					int endBri = tf.dims[0];
					forwardSlice(params, as, t, bparm, startI, endI, startBri, endBri, dmax, rmax);
				}else {
					AtomicInteger done = new AtomicInteger();
					int neededWorkers = Math.min(par, taskCount);
					ArrayList<Future<?>> workers = new ArrayList<>(neededWorkers);
					try {
						for(int i=0; i<neededWorkers; ++i) {
							workers.add(parallelizer.submit(()->{
								while(true) {
									int task = done.getAndAdd(1);
									if(task>=taskCount)
										return;
									int startI = task * ch / taskCount;
									int endI = (task+1) * ch / taskCount;
									int startBri = 0;
									int endBri = tf.dims[0];
									forwardSlice(params, as, t, bparm, startI, endI, startBri, endBri, dmax, rmax);
								}
							}));
						}
						Module.joinAll(workers);

					}finally {
						for(Future<?> f: workers)
							f.cancel(true);
					}

				}
			}else {
				int maxPar = Math.max(ch, as.length);

				int taskCount = Math.min(par*4, maxPar);
				if(par==1 || taskCount==1) {
					int startI = 0;
					int endI = ch;
					int startA = 0;
					int endA = as.length;
					int startBri = 0;
					int endBri = tf.dims[0];

					forwardInferenceSlice(params, as, t, startI, endI, startA, endA, startBri, endBri);
				}else{
					AtomicInteger done = new AtomicInteger();
					int workersNeeded = Math.min(par, maxPar);
					ArrayList<Future<?>> workers = new ArrayList<>(workersNeeded);
					try {
						if(maxPar==as.length) {
							for(int i=0; i<workersNeeded; ++i) {
								workers.add(parallelizer.submit(()->{
									while(true) {
										int task = done.getAndAdd(1);
										if(task>=taskCount)
											return;
										int startA = task * as.length / taskCount;
										int endA = (task+1) * as.length / taskCount;
										int startI = 0;
										int endI = ch;
										int startBri = 0;
										int endBri = tf.dims[0];
										forwardInferenceSlice(params, as, t, startI, endI, startA, endA, startBri, endBri);
									}
								}));
							}
						}else {
							for(int i=0; i<workersNeeded; ++i) {
								workers.add(parallelizer.submit(()->{
									while(true) {
										int task = done.getAndAdd(1);
										if(task>=taskCount)
											return;
										int startI = task * ch / taskCount;
										int endI = (task+1) * ch / taskCount;
										int startA = 0;
										int endA = as.length;
										int startBri = 0;
										int endBri = tf.dims[0];
										forwardInferenceSlice(params, as, t, startI, endI, startA, endA, startBri, endBri);
									}
								}));
							}
						}
						Module.joinAll(workers);

					}finally {
						for(Future<?> f: workers)
							f.cancel(true);
					}

				}


			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}

	}
	private void forwardInferenceSlice(ParamSet params, ActivationBatch as, int t, 
			int startI, int endI, int startA, int endA, int startBri, int endBri) {
		for(int bi = startA; bi<endA; ++bi) {
			ActivationSeq b = as.a[bi];
			if(b==null)
				continue;
			for(int i=startI; i<endI; ++i) {
				ActivationSet a = b.get(t);
				float mean = params.get(this.runningMean, i);
				float sdev = params.get(this.runningSdev, i);
				float addVal=hasAdd?params.get(add, i):0;
				float multVal=params.get(mult, i);
				//						if(i==0) {
				//							System.out.println("mean = "+mean+", var = "+var);
				//							System.out.println("add = "+addVal+", mult = "+multVal);
				//						}
				//						System.out.println(" (x + "+mShift+") * "+vScale+" * "+multVal+" + "+addVal);

				float scal = 1/sdev;
				for(int bri = startBri; bri<endBri; bri++) {
					int index = tf.index(bri, i);
					float d = (a.get(in, index) - mean)*scal;
					float v = d*multVal + addVal;
					a.add(out, index, v);
				}
			}
		}
	}
	private void forwardSlice(ParamSet params, ActivationBatch as, int t, ActivationSet bparm, int startI, int endI,
			int startBri, int endBri, float dmax, float rmax) {
		for(int i=startI; i<endI; ++i) {
			float mean=0;
			int count=0;
			for(ActivationSeq b: as) {
				if(b==null)
					continue;
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);
					mean+=b.get(t).get(in, index);
					count++;
				}
			}
			mean/=count;
			float var = 0;
			for(ActivationSeq b: as) {
				if(b==null)
					continue;
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);

					float d = b.get(t).get(in, index) - mean;
					var += d*d;
				}
			}
			var/=count;
			float sdev=(float) Math.sqrt(var+epsilon);
			float scal = 1/sdev;
			float rmean=params.get(runningMean, i);
			float rsdev=params.get(runningSdev, i);

			float r=Math.min(rmax, Math.max(1/rmax, sdev/rsdev));
			float d=Math.min(dmax, Math.max(-dmax, (mean-rmean)/rsdev));
			{
				bparm.set(this.mean, i, mean);
				bparm.set(this.sdev, i, sdev);
			}

			float addVal=hasAdd?params.get(add, i):0;
			float multVal=params.get(mult, i);
			//					if(i==0) {
			//						System.out.println("mean = "+mean+", var = "+var);
			//						System.out.println("add = "+addVal+", mult = "+multVal);
			//					}
			//					System.out.println(" (x - "+mean+") * "+scal+" * "+multVal+" + "+addVal);
			for(ActivationSeq b: as) {
				if(b==null)
					continue;
				ActivationSet a = b.get(t);
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);
					float dv = (a.get(in, index) - mean)*scal*r+d;
					float v = dv*multVal + addVal;
					a.add(out, index, v);
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		//		float dmax=dmaxSchedule.apply(age);
		float rmax=rmaxSchedule.apply(age);

		if(inst==null) {			
			int ch = tf.channels();
			int taskCount = Math.min(par*4, ch);
			if(par==1 || taskCount==1) {
				int startI = 0;
				int endI = ch;
				int startBri = 0;
				int endBri = tf.dims[0];

				backpropSlice(params, as, errors, t, rmax, startI, endI, startBri, endBri);
			}else {
				AtomicInteger done = new AtomicInteger();
				int workersNeeded = Math.min(par, taskCount);
				ArrayList<Future<?>> workers = new ArrayList<>(workersNeeded);
				try {
					for(int i=0; i<workersNeeded; ++i) {
						workers.add(parallelizer.submit(()->{
							while(true) {
								int task = done.getAndAdd(1);
								if(task>=taskCount)
									return;
								int startI = task * ch / taskCount;
								int endI = (task+1) * ch / taskCount;
								int startBri = 0;
								int endBri = tf.dims[0];

								backpropSlice(params, as, errors, t, rmax, startI, endI, startBri, endBri);
							}
						}));
					}
					Module.joinAll(workers);

				}finally {
					for(Future<?> f: workers)
						f.cancel(true);
				}

			}

		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}
	}
	private void backpropSlice(ParamSet params, ActivationBatch as, ActivationBatch errors, int t, float rmax,
			int startI, int endI,
			int startBri, int endBri) {
		for(int i=startI; i<endI; ++i) {

			float mean = as.batchParams.get(t).get(this.mean, i);
			float sdev = as.batchParams.get(t).get(this.sdev, i);
			float scal = 1/sdev;
			//				float rmean=params.get(runningMean, i);
			float rsdev=params.get(runningSdev, i);

			float r=Math.min(rmax, Math.max(1/rmax, sdev/rsdev));
			//				float d=Math.min(dmax, Math.max(-dmax, (mean-rmean)/rsdev));

			//				float addVal=params.get(add, i);
			float multVal=params.get(mult, i);
			int count=0;
			float scalErr=0;

			for(int b=0; b<as.length; ++b) {
				ActivationSeq es=errors.a[b];
				if(es==null)
					continue;
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);

					count++;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float me=e.get(out, index)*multVal;
					float scalInp = (a.get(in, index)-mean);
					scalErr+=me*scalInp;
				}


			}
			//			float scal = 1/Math.sqrt(var+epsilon);
			//			ð var = -0.5*(var+epsilon)^1.5 * ð scal;
			float varErr = scalErr * -0.5f / (sdev*sdev*sdev);
			varErr/=count;
			float meanErr=0;
			for(int b=0; b<as.length; ++b) {
				ActivationSeq es=errors.a[b];
				if(es==null)
					continue;
				ActivationSet a = as.a[b].get(t);
				ActivationSet e = es.get(t);
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);

					float scalInp = a.get(in, index)-mean;
					float me=e.get(out, index)*multVal*r;
					me *= scal;
					me += 2 * scalInp * varErr;
					meanErr += -me;
				}
			}
			meanErr/=count;
			for(int b=0; b<as.length; ++b) {
				ActivationSeq es=errors.a[b];
				if(es==null)
					continue;
				ActivationSet a = as.a[b].get(t);
				ActivationSet e = es.get(t);
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);

					float scalInp = a.get(in, index)-mean;
					float me=e.get(out, index)*multVal*r;
					me *= scal;
					me += 2 * scalInp * varErr;
					me += meanErr;
					e.add(in, index, me);
				}
			}
		}
	}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		float dmax=dmaxSchedule.apply(age);
		float rmax=rmaxSchedule.apply(age);
		if(inst==null) {
			int ch = tf.channels();
			int taskCount = Math.min(par*4, ch);
			if(par==1 || taskCount==1) {
				int startI = 0;
				int endI = ch;
				int startBri = 0;
				int endBri = tf.dims[0];

				gradientSlice(params, as, errors, gradients, t, dmax, rmax, startI, endI, startBri, endBri);
			}else {
				AtomicInteger done = new AtomicInteger();
				int workersNeeded = Math.min(par, taskCount);
				ArrayList<Future<?>> workers = new ArrayList<>(workersNeeded);
				try {
					for(int i=0; i<workersNeeded; ++i) {
						workers.add(parallelizer.submit(()->{
							while(true) {
								int task = done.getAndAdd(1);
								if(task>=taskCount)
									return;
								int startI = task * ch / taskCount;
								int endI = (task+1) * ch / taskCount;
								int startBri = 0;
								int endBri = tf.dims[0];

								gradientSlice(params, as, errors, gradients, t, dmax, rmax, startI, endI, startBri, endBri);
							}
						}));
					}
					Module.joinAll(workers);

				}finally {
					for(Future<?> f: workers)
						f.cancel(true);
				}

			}

		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}	
	}
	private void gradientSlice(ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients, int t,
			float dmax, float rmax, int startI, int endI,
			int startBri, int endBri) {
		ActivationSet bp = as.batchParams.get(t);
		for(int i=startI; i<endI; ++i) {
			float mean = bp.get(this.mean, i);
			float sdev = bp.get(this.sdev, i);
			float scal = 1/sdev;
			float rmean=params.get(runningMean, i);
			float rsdev=params.get(runningSdev, i);

			float r=Math.min(rmax, Math.max(1/rmax, sdev/rsdev));
			float d=Math.min(dmax, Math.max(-dmax, (mean-rmean)/rsdev));

			//				int count=0;

			float addErr=0;
			float multErr=0;
			for(int b=0; b<as.length; ++b) {
				ActivationSeq es=errors.a[b];
				if(es==null)
					continue;
				//					count++;
				ActivationSet a = as.a[b].get(t);
				ActivationSet e = es.get(t);
				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);

					float oe=e.get(out, index);
					addErr += oe;
					float reparin = (a.get(in, index)-mean)*scal*r+d;
					multErr += reparin * oe;
				}


			}
			if(hasAdd)
				gradients.add(add, i, addErr);
			gradients.add(mult, i, multErr);
			//				gradients.set(runningMean, i, 0);
			//				gradients.add(runningVar, i, 0);
		}
	}
	@Override
	public void dontComputeInPhase(String phase) {

	}
	@Override
	public void initParams(ParamSet p) {
		age=0;
		int ch = tf.channels();
		for(int i=0; i<ch; ++i) {
			if(hasAdd)
				p.set(add, i, 0);
			p.set(mult, i, 1);
			p.set(runningMean, i, 0);
			p.set(runningSdev, i, 1);
		}
	}

	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
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
	public void updateStatistics(ActivationSeq stat, ParamSet params, Function<Integer, Float> owt_fun, float[] weight, int tMin) {
		float owt=owt_fun.apply(age++);
		System.out.print("");
		int ch = tf.channels();
		for(int i=0; i<ch; ++i) {
			float smean, ssdev, swt;
			smean=params.get(runningMean, i)*owt;
			ssdev=params.get(runningSdev, i)*owt;
			swt=owt;
			for(int ts=0; ts<weight.length; ++ts) {
				int t = ts + tMin;
				float df=weight[ts];
				swt+=df;
				smean += df*stat.get(t).get(mean, i);
				ssdev += df*stat.get(t).get(sdev, i);
			}
			float mv = smean/swt;
			float vv = ssdev/swt;
			params.set(runningMean, i, mv);
			params.set(runningSdev, i, vv);
		}
	}
	public ParamBlock getAdd() {
		return add;
	}
	public ParamBlock getMult() {
		return mult;
	}
	public ParamBlock getRunningMean() {
		return runningMean;
	}
	public ParamBlock getRunningVar() {
		return runningSdev;
	}
	@Override
	public ParamBlock getBias() {
		return add;
	}
	@Override
	public ParamBlock getScale() {
		return mult;
	}
	public BatchReNorm parallel(int threads) {
		if(threads<1)
			throw new IllegalArgumentException();
		par = threads;
		return this;
	}
}
