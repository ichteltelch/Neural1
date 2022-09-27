package org.siquod.ml.neural1.net;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;

import org.siquod.ml.data.TrainingBatchCursor;
import org.siquod.ml.data.Whitener;
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
import org.siquod.ml.neural1.optimizers.Rprop;
import org.siquod.ml.neural1.optimizers.Updater;

public class FeedForward {
	public final Interface in, out, target, loss;
	InOutModule net;
	LossLayer lossLayer;
	int paramCount, activationCount, evalActivationCount;
	int decisionCount, dropoutCount;
	private InterfaceAllocator ba;
	public class Eval{
		{
			init();
		}
		ParamSet ps;
		public final ActivationBatch ass= new ActivationBatch(1, 1, ia, ba);
		ActivationSeq as=ass.a[0];
		ActivationSet a=as.get(0);
		float[] whitenedInputs = new float[in.count];
		private Whitener inputWhitener;
		public Eval(ParamSet ps){
			this.ps=ps;
			if(ps.size()!=paramCount)
				throw new IllegalArgumentException("Param set has wrong size");

		}
		public void eval(float[] input, float[] output){


			a.clear();
			//			as.clearLifeInterfaces(true);


			net.initializeRun(ass, false);
			a.set(in, applyWhitener(input));
			net.forward(ForwardPhase.TESTING, ps, ass, 0, null);
			a.get(out, output);

			//			lossLayer.forward(ForwardPhase.TRAINING, ps, ass, 0, null);
			//			double lossVal =a.get(loss, 0);
			//			//			System.out.println(lossVal);
			//			return lossVal;
		}
		private float[] applyWhitener(float[] input) {
			if(inputWhitener==null)
				return input;
			inputWhitener.whiten(input, whitenedInputs);
			return whitenedInputs;
		}
		public String showWeights() {
			StringBuilder r=new StringBuilder(); 
			for(Module n: net.deepSubmodules())
				if(n instanceof Dense)
					r.append(((Dense) n).showParams(ps));
			return r.toString();
		}
		public double computeLoss(TrainingBatchCursor data, int bs) {
			return computeLoss(data, bs, false);
		}
		public double computeLoss(TrainingBatchCursor data, int bs, boolean includeRegLoss) {
			ActivationBatch ass = new ActivationBatch(bs, 1, ia, ba);
			return computeLoss(data, ass, includeRegLoss, null);
		}
		public double computeLoss(TrainingBatchCursor data, int bs, boolean includeRegLoss, Consumer<? super ActivationBatch> afterBatch) {
			ActivationBatch ass = new ActivationBatch(bs, 1, ia, ba);
			return computeLoss(data, ass, includeRegLoss, afterBatch);
		}
		public double computeLoss(TrainingBatchCursor data, ActivationBatch ass, boolean includeRegLoss, Consumer<? super ActivationBatch> afterBatch) {
			int bs=ass.a.length;
			double[] inBuffer = new double[data.inputCount()];
			double[] outBuffer = new double[data.outputCount()];
			float[] inBufferFloat = new float[inBuffer.length];
			float[] outBufferFloat = new float[outBuffer.length];
			data.reset();
			double lossSum=0;
			int n=0;
			int bi = 0;
			net.initializeRun(ass, false);

			while(!data.isFinished()) {
				{
					data.giveInputs(inBuffer);
					data.giveOutputs(outBuffer);
					for(int j=0; j<inBuffer.length; ++j)
						inBufferFloat[j]=(float)inBuffer[j];
					for(int j=0; j<outBuffer.length; ++j)
						outBufferFloat[j]=(float)outBuffer[j];
					ActivationSeq as=ass.a[bi];
					ActivationSet a=as.get(0);
					a.clear();
					a.set(in, applyWhitener(inBufferFloat));
					a.set(target, outBufferFloat);
					data.next();
					++bi;
				}
				if(bi>=bs || data.isFinished()) {
					ass.length=bi;
					net.forward(ForwardPhase.TESTING, ps, ass, 0, null);
					lossLayer.forward(ForwardPhase.TESTING, ps, ass, 0, null);
					//					System.out.println(lossVal);
					for(bi=0; bi<ass.length; ++bi) {
						ActivationSeq as=ass.a[bi];
						ActivationSet a=as.get(0);
						double lossVal = a.get(loss, 0);
						if(includeRegLoss) lossVal += a.get(loss, 1);
						lossSum+=lossVal;
					}
					n+=bi;
					bi=0;
					if(afterBatch!=null)
						afterBatch.accept(ass);
					//					net.initializeRun(ass, false);
				}




			}
			return lossSum/n;
		}
		public int[][] computeConfusion(TrainingBatchCursor data, int bs) {
			int[][] ret = new int[data.outputCount()][data.outputCount()];
			double[] inBuffer = new double[data.inputCount()];
			double[] outBuffer = new double[data.outputCount()];
			int[] correctOutputs = new int[bs];
			float[] inBufferFloat = new float[inBuffer.length];
			float[] outBufferFloat = new float[outBuffer.length];
			data.reset();
			ActivationBatch ass = new ActivationBatch(bs, 1, ia, ba);
			int bi = 0;
			net.initializeRun(ass, false);

			while(!data.isFinished()) {
				{
					data.giveInputs(inBuffer);
					data.giveOutputs(outBuffer);
					int correctOutput = 0;
					{
						double best = Double.NEGATIVE_INFINITY;
						for(int i=0; i<outBuffer.length; ++i) {
							double v = outBuffer[i];
							if(v>best) {
								correctOutput = i;
								best = v;
							}
						}
					}
					correctOutputs[bi]=correctOutput;
					for(int j=0; j<inBuffer.length; ++j)
						inBufferFloat[j]=(float)inBuffer[j];
					ActivationSeq as=ass.a[bi];
					ActivationSet a=as.get(0);

					a.clear();
					a.set(in, applyWhitener(inBufferFloat));
					data.next();
					++bi;
				}
				if(bi>=bs || data.isFinished()) {
					ass.length = bi;
					net.forward(ForwardPhase.TESTING, ps, ass, 0, null);
					for(bi=0; bi<ass.length; ++bi) {
						ActivationSeq as=ass.a[bi];
						ActivationSet a=as.get(0);
						a.get(out, outBufferFloat);
						int correctOutput = correctOutputs[bi];
						int givenOutput = 0;
						{
							double best = Double.NEGATIVE_INFINITY;
							for(int i=0; i<outBufferFloat.length; ++i) {
								double v = outBufferFloat[i];
								if(v>best) {
									givenOutput = i;
									best = v;
								}
							}
						}
						++ret[correctOutput][givenOutput];
						net.initializeRun(ass, false);
					}
					bi=0;
				}
			}
			return ret;
		}
		public Eval inputWhitener(Whitener whitener) {
			inputWhitener = whitener;
			return this;
		}
		public ParamSet getParams() {
			return ps;
		}
	}
	public static String confusionMatrixToString(int[][] mat) {
		StringBuilder t = new StringBuilder();
		StringBuilder sb = new StringBuilder();
		int[] columnWidth = new int[mat[0].length];
		for(int[] row: mat) {
			for(int i=0; i<row.length; ++i) {
				int l = t.append(row[i]).length();
				t.delete(0, l);
				columnWidth[i] = Math.max(columnWidth[i], l);
			}
		}
		for(int[] row: mat) {
			for(int i=0; i<row.length; ++i) {
				int l = t.append(row[i]).length();
				t.delete(0, l);
				for(int p=columnWidth[i] - l +1; p>=0; --p)
					sb.append(' ');
				sb.append(row[i]);
			}
			sb.append('\n');
		}

		return sb.toString();
	}


	public class NaiveTrainer implements Cloneable{
		{
			init();
		}	
		ParamSet ps=new ParamSet(paramCount);
		ParamSet lrm=new ParamSet(paramCount);
		{
			Arrays.fill(lrm.value, 1);
			net.setLearnRateMultipliers(lrm);
		}
		ParamSet grad=new ParamSet(paramCount);

		//		Updater u=new Rprop();
		Updater u;//=new SGD();
		ActivationBatch ass, ess;
		//		ActivationSeq as=ia.makeSeq(1).init().entire();
		//		ActivationSet a=as.get(0);
		//		ActivationSeq es=ia.makeSeq(1).init().entire();
		//		ActivationSet e=es.get(0);
		public double learnRate=.01;
		public float globReg=1;
		int batchCounter=0;
		int currentBatchSize=0;
		private float[] imp;
		public float batchReNormWeight = 0.1f;;		
		@Override
		public NaiveTrainer clone() {
			NaiveTrainer r;
			try {
				r = (NaiveTrainer) super.clone();
			} catch (CloneNotSupportedException e) {
				e.printStackTrace();
				return null;
			}
			r.ps=ps.clone();
			r.grad=ps.clone();
			r.ass=ass.clone();
			r.ess=ess.clone();
			r.u=u.clone();
			r.imp=imp.clone();
			return r;
		}
		public NaiveTrainer(int batchSize, Updater updater) {
			ass=new ActivationBatch(batchSize, 1, ia, ba);
			ess=new ActivationBatch(batchSize, 1, ia, ba);
			imp=new float[batchSize];
			u = updater==null?new Rprop():updater;
		}

		public void epoch(TrainingBatchCursor data, int batchSize) {
			data.reset();
			if(batchSize<=0)
				batchSize=imp.length;
			double[] inBuffer = new double[data.inputCount()];
			double[] outBuffer = new double[data.inputCount()];
			float[] inBufferFloat = new float[inBuffer.length];
			float[] outBufferFloat = new float[outBuffer.length];
			//			System.out.println("Training for one epoch!");
			boolean last=false;
			@SuppressWarnings("unused")
			int count=0;
			while(!last) {
				for(currentBatchSize=0; currentBatchSize<batchSize; ++currentBatchSize, ++count) {
					if(data.isFinished()) {
						//						if(true) {
						//							currentBatchSize=0;
						//							return;
						//						}
						data.reset();
						last=true;
					}
					data.giveInputs(inBuffer);
					data.giveOutputs(outBuffer);
					for(int j=0; j<inBuffer.length; ++j)
						inBufferFloat[j]=(float)inBuffer[j];
					for(int j=0; j<outBuffer.length; ++j)
						outBufferFloat[j]=(float)outBuffer[j];
					ActivationSet a = ass.a[currentBatchSize].get(0);
					a.clear();
					a.set(in, inBufferFloat);
					a.set(target, outBufferFloat);
					imp[currentBatchSize]=(float) data.getWeight();
					data.next();
				}
				//				System.out.println("Consume next batch: "+currentBatchSize+" samples");
				//				System.out.println("processed samples: "+count);

				endBatch();
				if(data.isFinished())
					break;

			}
		}

		public void addToBatch(float[] input, float[] targ, double importance){
			ActivationSet a = ass.a[currentBatchSize].get(0);
			a.clear();
			a.set(in, input);
			a.set(target, targ);
			imp[currentBatchSize]=(float) importance;
			currentBatchSize++;
		}

		@SuppressWarnings("unused")
		private void nop() {			
		}

		public Eval getEvaluator(boolean copyParams) {
			return new Eval(copyParams?ps.clone():ps);
		}
		public double endBatch() {
			if(currentBatchSize==0)
				return 1e100;
			//			if(currentBatchSize!=imp.length)
			//				throw new IllegalStateException("The batch is not yet full");

			ass.length = ess.length = currentBatchSize;


			net.initializeRun(ass, true);
			lossLayer.initializeRun(ass, true);

			net.forward(ForwardPhase.TRAINING, ps, ass, 0, null);
			lossLayer.forward(ForwardPhase.TRAINING, ps, ass, 0, null);

			double ret = 0;
			for(ActivationSeq as: ass.a)
				ret += as.get(0).get(loss, 0);
			ret /= currentBatchSize;
			for(int i=0; i<currentBatchSize; ++i) {
				ActivationSeq es = ess.a[i];
				ActivationSet e = es.get(0);
				e.clear();
				e.add(loss, 0, imp[i]);
				e.add(loss, 1, imp[i]);
			}
			for(int i=currentBatchSize; i<imp.length; ++i)
				ess.a[i].get(0).clear();
			lossLayer.backprop("training", ps, ass, ess, 0, null);
			net.backprop("training", ps, ass, ess, 0, null);
			lossLayer.gradients("training", ps, ass, ess, grad, 0, null);
			net.gradients("training", ps, ass, ess, grad, 0, null);
			Function<Integer, Float> owt = batchCounter -> (float)Math.log(batchCounter+1);
			float[] weight = {batchReNormWeight };


			//			ass.clear();
			//			net.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, ass, 0, null);
			//			lossLayer.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, ass, 0, null);
			lossLayer.updateStatistics(ass.batchParams, ps, owt, weight, 0);
			net.updateStatistics(ass.batchParams, ps, owt, weight, 0);


			net.regularize("training", ps, grad, globReg*(float)currentBatchSize);

			u.apply(ps, grad, lrm, (float)learnRate, currentBatchSize);
			grad.clear();
			currentBatchSize=0;
			batchCounter++;
			return ret;
		}
		public NaiveTrainer initSmallWeights(double d) {
			return initSmallWeights(d, new Random());
		}
		public NaiveTrainer initSmallWeights(double d, Random r) {
			for(int i=0; i<ps.size(); ++i)
				ps.set(i, d*(r.nextDouble()-0.5));
			net.initParams(ps);
			lossLayer.initParams(ps);
			return this;
		}
		public ParamSet getParams() {
			return ps;
		}
		public int currentBatchSize() {
			return currentBatchSize;
		}
	}


	public FeedForward(InOutModule net, LossLayer lossLayer, int inw, int outw){
		this(net, lossLayer, new TensorFormat(inw), new TensorFormat(outw));
	}
	public FeedForward(InOutModule net, LossLayer lossLayer, TensorFormat inw, TensorFormat outw){
		this.net=net;
		this.lossLayer=lossLayer;
		in=new Interface("input", inw);
		out=new Interface("output", outw);
		target=new Interface("target", outw);
		loss=new Interface("loss", new TensorFormat(2));
		//		init();
	}
	public FeedForward(InOutModule net, LossLayer lossLayer, int inw, int outw, int tw){
		this(net, lossLayer, new TensorFormat(inw), new TensorFormat(outw), new TensorFormat(tw));
	}
	public FeedForward(InOutModule net, LossLayer lossLayer, TensorFormat inw, TensorFormat outw, TensorFormat tw){
		this.net=net;
		this.lossLayer=lossLayer;
		in=new Interface("input", inw);
		out=new Interface("output", outw);
		target=new Interface("target", tw);
		loss=new Interface("loss", new TensorFormat(2));
		//		init();
	}
	InterfaceAllocator ia;
	boolean inited = false;
	public FeedForward init() {
		if(inited)
			return this;
		inited = true;
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
		return this;
	}
	public NaiveTrainer getNaiveTrainer(int bs, Updater u) {
		return new NaiveTrainer(bs, u);
	}

}
