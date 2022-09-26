package org.siquod.ml.neural1.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

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

public class BatchlessNorm extends AbstractBatchNorm{

	private Interface in;
	private Interface out;
	private Interface loss;
	ParamBlock mean, sd, add, mult;
	int age=0;
	boolean hasAdd=true;
	boolean broadcast;
	int par=1;
	TensorFormat tf;
	@Override TensorFormat tf() {return tf;}
	Supplier<? extends Interface> lossSupplier;
	double gradmul=defaultGradMult;
	double lr = defaultLrMult;
	@Override
	public float learnRateMultiplier() {return (float)lr;}


	public BatchlessNorm(boolean hasAdd, boolean broadcast, Supplier<? extends Interface> lossSupplier) {
		this.hasAdd=hasAdd;
		this.broadcast=broadcast;
		this.lossSupplier = lossSupplier;
	}
	public BatchlessNorm(Supplier<? extends Interface> lossSupplier) {
		this(true, true, lossSupplier);
	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		loss = lossSupplier.get();
		lossSupplier=null;
		if(!in.tf.equals(out.tf))
			throw new IllegalArgumentException("input and output layer must be of the same size");
		if(broadcast) {
			tf=in.tf.to2D();		
		}else{
			tf = new TensorFormat(1, in.count);
		}
	}

	@Override
	public void allocate(ParamAllocator ia) {
		int ch = tf.channels();
		if(hasAdd) add = ia.allocate(new ParamBlock("add", ch));
		mult = ia.allocate(new ParamBlock("mult", ch));
		mean = ia.allocate(new ParamBlock("mean", ch));
		sd = ia.allocate(new ParamBlock("sdev", ch));
	}

	@Override
	public void share(ParamBlocks ps) {
		int ch = tf.channels();
		if(hasAdd) add = ps.get("add", ch);
		mult = ps.get("mult", ch);		
		mean = ps.get("mean", ch);
		sd = ps.get("sd", ch);		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("BatchReNorm");
		if(hasAdd) ret.add(add);
		ret.add(mult);
		ret.add(mean);
		ret.add(sd);
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(inst==null) {
			int ch = tf.channels();
			if(training != ForwardPhase.TESTING) {


				int taskCount = Math.min(par*4, ch);
				if(par==1 || taskCount==1) {
					int startA = 0;
					int endA = as.length;
					int startI = 0;
					int endI = ch;
					int startBri = 0;
					int endBri = tf.dims[0];
					forwardSlice(params, as, t, startI, endI, startA, endA, startBri, endBri, false);
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
									int startA = 0;
									int endA = as.length;
									int startI = task * ch / taskCount;
									int endI = (task+1) * ch / taskCount;
									int startBri = 0;
									int endBri = tf.dims[0];
									forwardSlice(params, as, t, startI, endI, startA, endA, startBri, endBri, false);
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

					forwardSlice(params, as, t, startI, endI, startA, endA, startBri, endBri, true);
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
										forwardSlice(params, as, t, startI, endI, startA, endA, startBri, endBri, true);
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
										forwardSlice(params, as, t, startI, endI, startA, endA, startBri, endBri, true);
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
	private void forwardSlice(ParamSet params, ActivationBatch as, int t, 
			int startI, int endI, int startA, int endA, int startBri, int endBri,
			boolean computeLoss) {
		for(int bi = startA; bi<endA; ++bi) {
			double lossContrib = 0;
			ActivationSeq b = as.a[bi];
			if(b==null)
				continue;
			ActivationSet a = b.get(t);
			for(int i=startI; i<endI; ++i) {
				float mean = params.get(this.mean, i);
				float sdev = params.get(this.sd, i);
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
					float activation = a.get(in, index);
					float normalized = (activation - mean)*scal;
					if(computeLoss)
						lossContrib += 0.5*normalized*normalized + Math.log(Math.abs(sdev));
					float denormalized = normalized*multVal + addVal;
					a.add(out, index, denormalized);
				}
				

			}
			if(computeLoss) {
			lossContrib *= gradmul;
			synchronized (loss) {
				a.add(loss, 1, (float)lossContrib);
			}
			}
		}
	}


	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		//		float dmax=dmaxSchedule.apply(age);

		if(inst==null) {			
			int ch = tf.channels();
			int taskCount = Math.min(par*4, ch);
			if(par==1 || taskCount==1) {
				int startI = 0;
				int endI = ch;
				int startBri = 0;
				int endBri = tf.dims[0];
				int startA = 0;
				int endA = as.length;

				backpropSlice(params, as, errors, t, startA, endA, startI, endI, startBri, endBri);
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
								int startA = 0;
								int endA = as.length;

								backpropSlice(params, as, errors, t, startA, endA, startI, endI, startBri, endBri);
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
	private void backpropSlice(ParamSet params, ActivationBatch as, ActivationBatch errors, int t,
			int startA, int endA, 
			int startI, int endI,
			int startBri, int endBri) {
		
		
		for(int bi = startA; bi<endA; ++bi) {
			ActivationSeq asbi = as.a[bi];
			if(asbi==null)
				continue;
			ActivationSeq esbi = errors.a[bi];
			if(esbi==null)
				continue;
//			ActivationSet a = asbi.get(t);
			ActivationSet e = esbi.get(t);
//			double ðloss_lr = e.get(loss, 0)*lr;
			for(int i=startI; i<endI; ++i) {
/**
 * ðin_actiavtion = ðout_activation · exp(-log_sd)
 */
				
//				float mean = params.get(this.mean, i);
				float sdev = params.get(this.sd, i);
//				float addVal=hasAdd?params.get(add, i):0;
				float multVal=params.get(mult, i);

				float scal = 1/sdev;
				for(int bri = startBri; bri<endBri; bri++) {
					int index = tf.index(bri, i);
//					float in_activation = a.get(in, index);
					float ðout_activation = e.get(out, index) * multVal;
					float ðin_activation = ðout_activation * scal;
					e.add(in, index, ðin_activation);
				}
				

			}

		}

	}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		if(inst==null) {
			int ch = tf.channels();
			int taskCount = Math.min(par*4, ch);
			if(par==1 || taskCount==1) {
				int startI = 0;
				int endI = ch;
				int startBri = 0;
				int endBri = tf.dims[0];

				gradientSlice(params, as, errors, gradients, t, startI, endI, startBri, endBri);
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

								gradientSlice(params, as, errors, gradients, t, startI, endI, startBri, endBri);
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
			int startI, int endI,
			int startBri, int endBri) {
		for(int i=startI; i<endI; ++i) {
			float mean = params.get(this.mean, i);
			float sdev = params.get(this.sd, i);
//			float mult=params.get(this.mult, i);
//			float add=this.hasAdd?params.get(this.add, i):0;
			float scal = 1/sdev;

			//				int count=0;

			double ðadd=0;
			double ðmult=0;
			double ðmean=0;
			double ðlog_sdev=0;
			for(int b=0; b<as.length; ++b) {
				ActivationSeq es=errors.a[b];
				if(es==null)
					continue;
				//					count++;
				ActivationSet a = as.a[b].get(t);
				ActivationSet e = es.get(t);
				double ðloss_lr = e.get(loss, 1)*gradmul;
				/**
				 * ðlog_sd = learn_rate_multiplier · ðloss · (1 - 2·(in_activation - mean)²·exp(-log_sd)²)
				 * ðmean = learn_rate_multiplier · ðloss · (- 2·(in_activation - mean) · exp(-log_sd)² )
				 */

				for(int bri = startBri; bri<endBri; ++bri) {
					int index = tf.index(bri, i);

					float ðout=e.get(out, index);
					ðadd += ðout;
					float in_activation = a.get(in, index);
					float shifted = in_activation - mean;
					float normalized = shifted*scal;
					ðmult += normalized * ðout;
					
					ðmean += ðloss_lr * (-normalized*scal);
					ðlog_sdev += ðloss_lr * (1 - normalized*normalized);
				}


			}
			if(hasAdd)
				gradients.add(this.add, i, ðadd);
			gradients.add(this.mult, i, ðmult);
			gradients.add(this.mean, i, ðmean);
			// inv_sd = exp(-log_sd)
//			double ðscal = ðlog_sdev * -scal;
			double ðsdev = ðlog_sdev * scal;;
			// log_sdev = log sdev
			// ðsdev = ðlog_sdev / sdev
			// scal = 1/sdev = exp (- log_sdev))
			// ðsdev = ðlog_sdev * exp (- log_sdev)) * -1/sdev =  ðlog_sdev * (1/sdev) * -1/sdev 
			gradients.add(this.sd, i, ðsdev);
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
			p.set(mean, i, 0);
			p.set(sd, i, 1);
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

	public ParamBlock getAdd() {
		return add;
	}
	public ParamBlock getMult() {
		return mult;
	}
	public BatchlessNorm learn_rate_multiplier(double lr) {
		this.lr=lr;
		return this;
	}
	@Override
	public ParamBlock getBias() {
		return add;
	}
	@Override
	public ParamBlock getScale() {
		return mult;
	}
	public BatchlessNorm parallel(int threads) {
		if(threads<1)
			throw new IllegalArgumentException();
		par = threads;
		return this;
	}
	
	@Override
	public ParamBlock sigmaStorage() {
		return sd;
	}
	@Override
	public ParamBlock meanStorage() {
		return mean;
	}
	@Override
	public double distributionMismatch(ActivationBatch as, ParamSet params, int t) {
		int startI = 0;
		int endI = tf.channels();
		int startA = 0;
		int endA = as.length;
		int startBri = 0;
		int endBri = tf.dims[0];
		double ret = 0;
		for(int bi = startA; bi<endA; ++bi) {
			double lossContrib = 0;
			ActivationSeq b = as.a[bi];
			if(b==null)
				continue;
			ActivationSet a = b.get(t);
			for(int i=startI; i<endI; ++i) {
				float mean = params.get(this.mean, i);
				float sdev = params.get(this.sd, i);
				//						if(i==0) {
				//							System.out.println("mean = "+mean+", var = "+var);
				//							System.out.println("add = "+addVal+", mult = "+multVal);
				//						}
				//						System.out.println(" (x + "+mShift+") * "+vScale+" * "+multVal+" + "+addVal);

				float scal = 1/sdev;
				double entropy = 0.5*(1-Math.log(scal*scal));
				for(int bri = startBri; bri<endBri; bri++) {
					int index = tf.index(bri, i);
					float activation = a.get(in, index);
					float normalized = (activation - mean)*scal;
					lossContrib += 0.5*normalized*normalized + Math.log(Math.abs(sdev))-entropy;
				}
				

			}
			ret += lossContrib;	
		}
		return ret;
	}
	
	@Override
	public double mapSigmaFromStorage(double s) {
		return s;
	}
	@Override
	public double mapSigmaToStorage(double s) {
		return s;
	}
}
