package org.siquod.ml.neural1.net;

import java.util.Arrays;
import java.util.BitSet;
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
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;
import org.siquod.ml.neural1.modules.Dense;
import org.siquod.ml.neural1.modules.InOutModule;
import org.siquod.ml.neural1.modules.loss.LossLayer;
import org.siquod.ml.neural1.optimizers.AmsGrad;
import org.siquod.ml.neural1.optimizers.Updater;

public class Recurrent {
	public final Interface in, out, target, loss;
	InOutModule net;
	LossLayer lossLayer;
	int paramCount, activationCount, evalActivationCount;
	int decisionCount, dropoutCount;
	private InterfaceAllocator ba;
	public class Eval{
		ParamSet ps;
		ActivationBatch ass;
		ActivationSeq as;
		int t=0;
		public Eval(int length, ParamSet ps){
			this.ps=ps;
			if(ps.size()!=paramCount)
				throw new IllegalArgumentException("Param set has wrong size");
			ass = new ActivationBatch(1, length, ia, ba);
			as=ass.a[0];
		}
		public void newRun() {
			net.initializeRun(ass, false);
			as.clear();
			t=0;

		}
		public void step(float[] input, float[] output){
			as.advance();
			ActivationSet a = as.get(t);
			a.set(in, input); 

			net.forward(ForwardPhase.TESTING, ps, ass, t, null);
			a.get(out, output);
			++t;
		}
		public String showWeights() {
			StringBuilder r=new StringBuilder(); 
			for(Module n: net.deepSubmodules())
				if(n instanceof Dense)
					r.append(((Dense) n).showParams(ps));
			return r.toString();
		}
	}
	public class NaiveTrainer{
		ParamSet ps=new ParamSet(paramCount);
		ParamSet grad=new ParamSet(paramCount);
		ParamSet lrm=new ParamSet(paramCount);
		{
		}
		//		Updater u=new Adam();
		//		Updater u=new Rprop();
		//		Updater u=new SGD();
		Updater u=new AmsGrad();
		ActivationBatch ass, ess;
		public double learnRate=.005;
		public float globReg=1;
		BitSet deadLayers=new BitSet();
		int length;
		public NaiveTrainer(int batchSize, int length) {
			setBatchSizeAndLength(batchSize, length);			
		}
		public void setLength(int length) {
			this.length=length;
			setBatchSize(imp.length);
		}
		public void setBatchSizeAndLength(int batchSize, int length) {
			this.length=length;
			setBatchSize(batchSize);
		}
		public void setBatchSize(int batchSize) {
			ass=ess=null;
			ass=new ActivationBatch(batchSize, length, ia, ba);
			ess=new ActivationBatch(batchSize, length, ia, ba);
			imp=new float[batchSize];
			minIg=length;
			ig=new int[batchSize];
		}
		int currentBatchSize=0;
		private float[] imp;
		int minIg;
		int[] ig;
		int batchCounter=0;
		float impsum=0;

		public void addToBatch(float[][] input, float[][] targ, double importance, int ignorePrefix){
			ActivationSeq as=ass.a[currentBatchSize];
			imp[currentBatchSize]=(float) importance;
			minIg=Math.min(minIg, ig[currentBatchSize]=ignorePrefix);
			for(int t=0; t<length; ++t) {
				ActivationSet a = as.get(t);
				a.clear();
				a.set(in, input[t]);
				if(t>=ignorePrefix)
					a.set(target, targ[t]);
			}
			currentBatchSize++;
		}
		//		public void determineStatistics(double[][] inputs) {
		//			ArrayList<ActivationSeq> al=new ArrayList<>(inputs.length);
		//			for(double[] input: inputs) {
		//				ActivationSeq as=ia.makeSeq(1).init().entire();
		//				as.get(0).set(in, input);
		//				al.add(as);
		//			}
		//			net.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, al, 0, null);
		//			lossLayer.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, al, 0, null);			
		//		}
		//		public void determineStatistics(List<double[]> inputs) {
		//			ArrayList<ActivationSeq> al=new ArrayList<>(inputs.size());
		//			for(double[] input: inputs) {
		//				ActivationSeq as=ia.makeSeq(1).init().entire();
		//				as.get(0).set(in, input);
		//				al.add(as);
		//			}
		//			net.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, al, 0, null);
		//			lossLayer.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, al, 0, null);			
		//		}
		void nop() {			
		}

		public Eval getEvaluator(int i, boolean copyParams) {
			return new Eval(i+1, copyParams?ps.clone():ps);
		}
		public double endBatch() {
			if(currentBatchSize==0)
				return 1e100;
			if(currentBatchSize!=imp.length)
				throw new IllegalStateException("The batch is not yet full");




			net.initializeRun(ass, true);
			lossLayer.initializeRun(ass, true);

			for(int t=0; t<length; ++t)
				net.forward(ForwardPhase.TRAINING, ps, ass, t, null);
			for(int t=minIg; t<length; ++t)
				lossLayer.forward(ForwardPhase.TRAINING, ps, ass, t, null);

			int count=0;
			double ret = 0;
			for(int i=0; i<currentBatchSize; ++i)
				for(int t=ig[i]; t<length; ++t, ++count)
					ret += ass.a[i].get(t).get(loss, 0);
			ret /= count;
			for(int i=0; i<currentBatchSize; ++i) {
				ActivationSeq es = ess.a[i];
				for(int t=0; t<length; ++t) {
					ActivationSet e = es.get(t);
					e.clear();
					if(t>=ig[i]) {
						e.add(loss, 0, imp[i]);
						impsum+=imp[i];
					}
				}
			}
			for(int t=length-1; t>=0; --t) {
				lossLayer.backprop("training", ps, ass, ess, t, null);
				net.backprop("training", ps, ass, ess, t, null);
			}
			for(int t=length-1; t>=0; --t) {			
				lossLayer.gradients("training", ps, ass, ess, grad, t, null);
				net.gradients("training", ps, ass, ess, grad, t, null);
			}

			Function<Integer, Float> owt = batchCounter -> (float)Math.sqrt(batchCounter);
			float[] weight = new float[length/2];
			Arrays.fill(weight, 1);

			lossLayer.updateStatistics(ass.batchParams, ps, owt, weight, length-weight.length);
			net.updateStatistics(ass.batchParams, ps, owt, weight, length-weight.length);


			currentBatchSize=0;
			minIg=length;
			return ret;
		}
		public void endMetaBatch() {
			net.regularize("training", ps, grad, globReg*impsum);

			grad.clip(5f);
			u.apply(ps, grad, lrm, (float)learnRate, impsum);
			grad.clear();
			impsum=0;
		}
		public NaiveTrainer initSmallWeights(double d) {
			Random rnd=new Random();
			for(int i=0; i<ps.size(); ++i) {
				ps.set(i, d*(rnd.nextDouble()-0.5));
				//				ps.set(i, d*rnd.nextGaussian());
			}
			net.initParams(ps);
			lossLayer.initParams(ps);
			return this;
		}
		public ParamSet getParams() {
			return ps;
		}
		public ParamSet gradientsOfBatch(double[] lossOut) {
			if(currentBatchSize==0)
				return null;
			if(currentBatchSize!=imp.length)
				throw new IllegalStateException("The batch is not yet full");




			net.initializeRun(ass, true);
			lossLayer.initializeRun(ass, true);

			for(int t=0; t<length; ++t)
				net.forward(ForwardPhase.TRAINING, ps, ass, t, null);
			for(int t=minIg; t<length; ++t)
				lossLayer.forward(ForwardPhase.TRAINING, ps, ass, t, null);

			int count=0;

			lossOut[0] = 0;
			for(int i=0; i<currentBatchSize; ++i)
				for(int t=ig[i]; t<length; ++t, ++count)
					lossOut[0] += ass.a[i].get(t).get(loss, 0);
			lossOut[0] /= count;

			for(int i=0; i<currentBatchSize; ++i) {
				ActivationSeq es = ess.a[i];
				for(int t=0; t<length; ++t) {
					ActivationSet e = es.get(t);
					e.clear();
					//					if(t==length-1)
					if(t>=ig[i])
						e.add(loss, 0, 1.0f/count);
				}
			}
			for(int t=length-1; t>=0; --t) {
				lossLayer.backprop("training", ps, ass, ess, t, null);
				net.backprop("training", ps, ass, ess, t, null);
			}
			for(int t=length-1; t>=0; --t) {			
				lossLayer.gradients("training", ps, ass, ess, grad, t, null);
				net.gradients("training", ps, ass, ess, grad, t, null);
			}
			ParamSet ret=grad.clone();
			grad.clear();
			currentBatchSize=0;
			minIg=length;
			return ret;

		}
		public double loss(ParamSet shifted) {
			if(currentBatchSize==0)
				return 1e100;
			if(currentBatchSize!=imp.length)
				throw new IllegalStateException("The batch is not yet full");




			//			net.initializeRun(ass, true);
			//			lossLayer.initializeRun(ass, true);

			for(int t=0; t<length; ++t)
				net.forward(ForwardPhase.TRAINING, shifted, ass, t, null);
			for(int t=minIg; t<length; ++t)
				lossLayer.forward(ForwardPhase.TRAINING, shifted, ass, t, null);

			int count=0;

			double ret = 0;
			for(int i=0; i<currentBatchSize; ++i)
				for(int t=ig[i]; t<length; ++t, ++count)
					ret  += ass.a[i].get(t).get(loss, 0);
			ret /= count;
			currentBatchSize=0;
			minIg=length;
			return ret;
		}
		public NaiveTrainer setUpdater(Updater nu) {
			u=nu;
			return this;
		}
	}


	public Recurrent(InOutModule net, LossLayer lossLayer, int inw, int outw){
		this(net, lossLayer, new TensorFormat(inw), new TensorFormat(outw));
	}
	public Recurrent(InOutModule net, LossLayer lossLayer, TensorFormat inw, TensorFormat outw){
		this.net=net;
		this.lossLayer=lossLayer;
		in=new Interface("input", inw);
		out=new Interface("output", outw);
		target=new Interface("target", outw);
		loss=new Interface("loss", new TensorFormat(1));
		init();
	}
	InterfaceAllocator ia;
	private void init() {
		ia=new InterfaceAllocator();
		ba=new InterfaceAllocator();
		ia.allocate(in);
		ia.allocate(out);
		net.allocate(ia, in.name, out.name);
		net.allocateStatistics(ba);
		evalActivationCount=ia.getCount();

		ia.allocate(target);
		ia.allocate(loss);
		lossLayer.allocate(ia, out.name, target.name, loss.name);
		lossLayer.allocateStatistics(ba);

		activationCount=ia.getCount();
		dropoutCount=ia.getDropoutCount();
		decisionCount=ia.getDecisionCount();
		ParamAllocator pa=new ParamAllocator();
		net.allocate(pa);
		lossLayer.allocate(pa);
		paramCount=pa.getCount();
		lossLayer.dontComputeInPhase("test");
		loss.dontComputeInPhase("test");
		//		Dependencies ds=new Dependencies();
		//		net.declareDependencies(ds);
		//		lossLayer.declareDependencies(ds);
		//		DependencyGraph gr=ds.createGraph(ia);
		//		evalPlan=gr.makePlan(0, true, "test");
		//		trainPlan=gr.makePlan(0, false, "training");
	}
	public NaiveTrainer getNaiveTrainer(int bs, int length) {
		return new NaiveTrainer(bs, length);
	}

}
