package org.siquod.neural1.net;

import java.util.Random;
import java.util.function.Function;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ActivationSeq;
import org.siquod.neural1.ActivationSet;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamSet;
import org.siquod.neural1.TensorFormat;
import org.siquod.neural1.data.TrainingBatchCursor;
import org.siquod.neural1.data.TrainingBatchCursor.RandomAccess;
import org.siquod.neural1.modules.Dense;
import org.siquod.neural1.modules.InOutModule;
import org.siquod.neural1.modules.loss.LossLayer;
import org.siquod.neural1.updaters.Rprop;
import org.siquod.neural1.updaters.Updater;

public class FeedForward {
	public final Interface in, out, target, loss;
	InOutModule net;
	LossLayer lossLayer;
	int paramCount, activationCount, evalActivationCount;
	int decisionCount, dropoutCount;
	private InterfaceAllocator ba;
	public class Eval{
		ParamSet ps;
		ActivationBatch ass= new ActivationBatch(1, 1, ia, ba);
		ActivationSeq as=ass.a[0];
		ActivationSet a=as.get(0);
		public Eval(ParamSet ps){
			this.ps=ps;
			if(ps.size()!=paramCount)
				throw new IllegalArgumentException("Param set has wrong size");
		}
		public double eval(float[] input, float[] output){
			a.clear();
			//			as.clearLifeInterfaces(true);

			net.initializeRun(ass, false);

			a.set(in, input);
			net.forward(ForwardPhase.TESTING, ps, ass, 0, null);
			a.get(out, output);
			lossLayer.forward(ForwardPhase.TRAINING, ps, ass, 0, null);
			double lossVal =a.get(loss, 0);
//			System.out.println(lossVal);
			return lossVal;
		}
		public String showWeights() {
			StringBuilder r=new StringBuilder(); 
			for(Module n: net.deepSubmodules())
				if(n instanceof Dense)
					r.append(((Dense) n).showParams(ps));
			return r.toString();
		}
		public double computeLoss(RandomAccess data) {
			double[] inBuffer = new double[data.inputCount()];
			double[] outBuffer = new double[data.inputCount()];
			float[] inBufferFloat = new float[inBuffer.length];
			float[] outBufferFloat = new float[outBuffer.length];
			data.reset();
			double lossSum=0;
			int n=0;
			while(!data.isFinished()) {
				data.giveInputs(inBuffer);
				data.giveOutputs(outBuffer);
				for(int j=0; j<inBuffer.length; ++j)
					inBufferFloat[j]=(float)inBuffer[j];
				for(int j=0; j<outBuffer.length; ++j)
					outBufferFloat[j]=(float)outBuffer[j];

				a.clear();
				net.initializeRun(ass, false);
				a.set(in, inBufferFloat);
				a.set(target, outBufferFloat);
				
				net.forward(ForwardPhase.TESTING, ps, ass, 0, null);
				lossLayer.forward(ForwardPhase.TESTING, ps, ass, 0, null);
				double lossVal =a.get(loss, 0);
//				System.out.println(lossVal);
				
				lossSum+=lossVal;
				n++;
				data.next();

			}
			return lossSum/n;
		}
	}
	public class NaiveTrainer implements Cloneable{
		ParamSet ps=new ParamSet(paramCount);
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
			int count=0;
			while(!last) {
				for(currentBatchSize=0; currentBatchSize<batchSize; ++currentBatchSize, ++count) {
					if(data.isFinished()) {
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
			}
			for(int i=currentBatchSize; i<imp.length; ++i)
				ess.a[i].get(0).clear();
			lossLayer.backprop("training", ps, ass, ess, 0, null);
			net.backprop("training", ps, ass, ess, 0, null);
			lossLayer.gradients("training", ps, ass, ess, grad, 0, null);
			net.gradients("training", ps, ass, ess, grad, 0, null);
			Function<Integer, Float> owt = batchCounter -> (float)Math.log(batchCounter+1);
			float[] weight = {.1f};


			//			ass.clear();
			//			net.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, ass, 0, null);
			//			lossLayer.forward(ForwardPhase.BATCHNORM_STATISTICS, ps, ass, 0, null);
			lossLayer.updateStatistics(ass.batchParams, ps, owt, weight, 0);
			net.updateStatistics(ass.batchParams, ps, owt, weight, 0);


			net.regularize("training", ps, grad, globReg*(float)currentBatchSize);

			u.apply(ps, grad, (float)learnRate, currentBatchSize);
			grad.clear();
			currentBatchSize=0;
			batchCounter++;
			return ret;
		}
		public NaiveTrainer initSmallWeights(double d) {
			for(int i=0; i<ps.size(); ++i)
				ps.set(i, d*(Math.random()-0.5));
			net.initParams(ps);
			lossLayer.initParams(ps);
			return this;
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
		loss=new Interface("loss", new TensorFormat(1));
		init();
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
	public NaiveTrainer getNaiveTrainer(int bs, Updater u) {
		return new NaiveTrainer(bs, u);
	}

}
