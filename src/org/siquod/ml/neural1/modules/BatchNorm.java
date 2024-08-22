package org.siquod.ml.neural1.modules;

import java.util.Collections;
import java.util.List;
import java.util.Random;
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

public class BatchNorm extends BatchNormoid{

	private static final float epsilon = 1e-10f;
	private Interface in;
	private Interface out;
	private Interface mean, var;
	boolean hasAdd=true;
	ParamBlock add, mult;
	TensorFormat tf;
	public BatchNorm(BatchNorm copyThis) {
		super(copyThis);
		this.in = copyThis.in;
        this.out = copyThis.out;
        this.mean = copyThis.mean;
        this.var = copyThis.var;
        this.hasAdd = copyThis.hasAdd;
        this.add = copyThis.add;
        this.mult = copyThis.mult;
        this.tf = copyThis.tf;
	}
	@Override
	public BatchNorm copy() {
		return this;
	}

	public BatchNorm(boolean hasAdd) {
		this.hasAdd=hasAdd;
	}
	public BatchNorm() {
		this(true);
	}
	@Override TensorFormat tf() {return tf;}

	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		if(in.count!=out.count)
			throw new IllegalArgumentException("input and output layer must be of the same size");
		tf = new TensorFormat(1, in.count);
	}

	@Override
	public void allocateStatistics(InterfaceAllocator ia) {
		ia.push(null);
		mean = ia.allocate(new Interface("mean", in.tf));
		var = ia.allocate(new Interface("var", in.tf));
		ia.pop();
	}
	@Override
	public void allocate(ParamAllocator ia) {
		if(hasAdd) add = ia.allocate(new ParamBlock("add", in.count));
		mult = ia.allocate(new ParamBlock("mult", in.count));
		runningMean = ia.allocate(new ParamBlock("mean", in.count));
		runningSdev = ia.allocate(new ParamBlock("sdev", in.count));
	}

	@Override
	public void share(ParamBlocks ps) {
		if(hasAdd) add = ps.get("add", in.count);
		mult = ps.get("mult", in.count);		
		runningMean = ps.get("mean", in.count);
		runningSdev = ps.get("sdev", in.count);		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("Conv1D");
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
			if(training != ForwardPhase.TESTING) {
				for(int i=0; i<in.count; ++i) {
					float mean=0;
					int count=0;
					for(ActivationSeq b: as) {
						if(b==null)
							continue;
						mean+=b.get(t).get(in, i);
						count++;
					}
					mean/=count;
					float var = 0;
					for(ActivationSeq b: as) {
						if(b==null)
							continue;
						float d = b.get(t).get(in, i) - mean;
						var += d*d;
					}
					var/=count-1;

					float scal = 1/(float) (Math.sqrt(var+epsilon));
					{
						bparm.set(this.mean, i, mean);
						bparm.set(this.var, i, var);
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
						float d = (a.get(in, i) - mean)*scal;
						if(drownout!=0)
							d = d * drownOutDataAmplitude + (float)(drowner.nextGaussian())*drownOutNoiseAmplitude;
						float v = d*multVal + addVal;
						a.add(out, i, v);
					}
				}
			}else {
				for(ActivationSeq b: as) {
					if(b==null)
						continue;
					++instanceCount;

					for(int i=0; i<in.count; ++i) {
						ActivationSet a = b.get(t);
						float mean = params.get(this.runningMean, i);
						float sdev = params.get(this.runningSdev, i);
						float a_in = a.get(in, i);
						if(finalizationMode==FinalizationMode.MEANS) {
							params.add(this.runningMean, i, a_in);
						}else if(finalizationMode==FinalizationMode.STANDARD_DEVIATIONS) {
							double dev = a_in - mean;
							params.add(this.runningSdev, i, dev*dev);
						}

						float addVal=hasAdd?params.get(add, i):0;
						float multVal=params.get(mult, i);
						//						if(i==0) {
						//							System.out.println("mean = "+mean+", var = "+var);
						//							System.out.println("add = "+addVal+", mult = "+multVal);
						//						}
						//						System.out.println(" (x + "+mShift+") * "+vScale+" * "+multVal+" + "+addVal);



						float scal = 1/(sdev+epsilon);

						float d = (a_in - mean)*scal;
						float v = d*multVal + addVal;
						a.add(out, i, v);
					}
				}
			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}

	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst==null) {
			for(int i=0; i<in.count; ++i) {

				float mean = as.batchParams.get(t).get(this.mean, i);
				float var = as.batchParams.get(t).get(this.var, i);
				float scal = 1/(float)Math.sqrt(var+epsilon);

				//				float addVal=params.get(add, i);
				float multVal=params.get(mult, i)*drownOutDataAmplitude;
				int count=0;
				float scalErr=0;

				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					count++;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float me=e.get(out, i)*multVal;
					float scalInp = (a.get(in, i)-mean);
					scalErr+=me*scalInp;


				}
				//			float scal = 1/Math.sqrt(var+epsilon);
				//			ð var = -0.5*(var+epsilon)^1.5 * ð scal;
				float varErr = scalErr * -0.5f / ((var+epsilon)*(float)Math.sqrt(var+epsilon));
				varErr/=count-1;
				float meanErr=0;
				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float scalInp = a.get(in, i)-mean;
					float me=e.get(out, i)*multVal;
					me *= scal;
					me += 2 * scalInp * varErr;
					meanErr += -me;
				}
				meanErr/=count;
				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float scalInp = a.get(in, i)-mean;
					float me=e.get(out, i)*multVal;
					me *= scal;
					me += 2 * scalInp * varErr;
					me += meanErr;
					e.add(in, i, me);
				}
			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}
	}
	public float learnRateMultiplier() {return (float)0;}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		if(inst==null) {
			for(int i=0; i<in.count; ++i) {

				float mean = as.batchParams.get(t).get(this.mean, i);
				float var = as.batchParams.get(t).get(this.var, i);
				float scal = 1/(float)Math.sqrt(var+epsilon);

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
					float oe=e.get(out, i);
					addErr += oe;
					float reparin = (a.get(in, i)-mean)*scal;
					multErr += reparin * oe;


				}
				if(hasAdd)
					gradients.add(add, i, addErr);
				gradients.add(mult, i, multErr);
				//				gradients.set(runningMean, i, 0);
				//				gradients.add(runningVar, i, 0);
			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}	
	}
	@Override
	public void dontComputeInPhase(String phase) {

	}
	@Override
	public void initParams(ParamSet p) {
		age=0;
		for(int i=0; i<in.count; ++i) {
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
	int age=0;
	@Override
	public void updateStatistics(ActivationSeq stat, ParamSet params, Function<Integer, Float> owt_fun, float[] weight, int tMin) {
		float owt=owt_fun.apply(age++);
		System.out.print("");
		for(int i=0; i<in.count; ++i) {
			float smean, ssdev, swt;
			smean=params.get(runningMean, i)*owt;
			ssdev=params.get(runningSdev, i)*owt;
			swt=owt;
			for(int ts=0; ts<weight.length; ++ts) {
				int t = ts + tMin;
				float df=weight[ts];
				swt+=df;
				smean += df*stat.get(t).get(mean, i);
				ssdev += df*Math.sqrt(stat.get(t).get(var, i));
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
	public ParamBlock getBias(){
		return add;
	}
	public ParamBlock getMult() {
		return mult;
	}
	public ParamBlock getRunningMean() {
		return runningMean;
	}
	public ParamBlock getRunningSdev() {
		return runningSdev;
	}
	@Override
	public ParamBlock getScale() {
		return mult;
	}
	@Override
	public double distributionMismatch(ActivationBatch as, ParamSet params, int t) {
		int startI = 0;
		int endI = in.count;
		int startA = 0;
		int endA = as.length;
		double ret = 0;
		for(int bi = startA; bi<endA; ++bi) {
			double lossContrib = 0;
			ActivationSeq b = as.a[bi];
			if(b==null)
				continue;
			ActivationSet a = b.get(t);
			for(int i=startI; i<endI; ++i) {
				float mean = params.get(this.runningMean, i);
				float sdev = params.get(this.runningSdev, i);

				float scal = 1/sdev;
				double entropy = 0.5*(1+Math.log(sdev*sdev));

				int index = i;
				float activation = a.get(in, index);
				float normalized = (activation - mean)*scal;
				lossContrib += 0.5*normalized*normalized + Math.log(sdev) -entropy;


			}
			ret += lossContrib;	
		}
		return ret;
	}
	Random drowner = new Random();
	double drownout=0;
	float drownOutNoiseAmplitude = 0;
	float drownOutDataAmplitude = 1;
	public BatchNorm drownout(double drown) {
		drownout=drown;
		drownOutNoiseAmplitude = (float) Math.sqrt(drown);
		drownOutDataAmplitude = (float) Math.sqrt(1-drown);
		return this;
	}
}
