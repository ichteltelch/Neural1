package org.siquod.ml.neural1.examples;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.ByteLookupTable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.function.ToDoubleFunction;

import javax.management.monitor.StringMonitorMBean;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.siquod.ml.data.ShuffledCursor;
import org.siquod.ml.data.TrainingBatchCursor;
import org.siquod.ml.data.TrainingBatchCursor.RandomAccess;
import org.siquod.ml.neural1.modules.AbstractBatchNorm;
import org.siquod.ml.neural1.modules.BatchNorm;
import org.siquod.ml.neural1.modules.BatchNormoid;
import org.siquod.ml.neural1.modules.BatchNormoid.FinalizationMode;
import org.siquod.ml.neural1.modules.BatchReNorm;
import org.siquod.ml.neural1.modules.BatchlessNorm;
import org.siquod.ml.neural1.modules.BatchlessNormLog;
import org.siquod.ml.neural1.modules.BatchlessNormInv;
import org.siquod.ml.neural1.modules.Copy;
import org.siquod.ml.neural1.modules.Dense;
import org.siquod.ml.neural1.modules.Dropout;
import org.siquod.ml.neural1.modules.InOutModule;
import org.siquod.ml.neural1.modules.Nonlin;
import org.siquod.ml.neural1.modules.StackModule;
import org.siquod.ml.neural1.modules.loss.SoftMaxNllLoss;
import org.siquod.ml.neural1.modules.regularizer.L2Reg;
import org.siquod.ml.neural1.modules.regularizer.Regularizer;
import org.siquod.ml.neural1.net.FeedForward;
import org.siquod.ml.neural1.neurons.Isrlu;
import org.siquod.ml.neural1.neurons.Neuron;
import org.siquod.ml.neural1.optimizers.Adam;
import org.siquod.ml.neural1.optimizers.AmsGrad;
import org.siquod.ml.neural1.optimizers.Rprop;
import org.siquod.ml.neural1.optimizers.SGD;


public class BatchlessTest2 implements Runnable{
	static enum BnType{
		NONE,
		BN,
		BRN,
		BLN,
		BLNLOG,
		BLNINV,
	}
	static int dataMultiplier = 10;
	//Example data sets
	static float[][][] data1;
	static float[][][] data2;
	static float[][][] data2Test;
	static float[][] sample;
	static boolean dlrc;

	static{
		data1=new float[3][100][2];
		float[] cx = {1,-.5f,-.5f};
		float[] cy = {0,(float) Math.sqrt(.75),(float) -Math.sqrt(.75)};
		Random rnd=new Random(12345);
		for(int i=0; i<data1.length; i++){
			for(int j=0; j<data1[i].length; j++){
				data1[i][j][0] = (float) (cx[i] + rnd.nextGaussian()*.5);
				data1[i][j][1] = (float) (cy[i] + rnd.nextGaussian()*.5);
			}
		}
		data2=new float[3][200*dataMultiplier][2];
		data2Test=new float[3][40*dataMultiplier][2];
		makeData2(data2, rnd);		
		makeData2(data2Test, rnd);	
		ArrayList<float[]> samp = new ArrayList<>();
		for(int sx = -100; sx<=100; sx+=10)
			for(int sy = -100; sy<=100; sy+=10)
				samp.add(new float[] {sx, sy});
		sample = samp.toArray(new float[samp.size()][]);
	}
	private static void makeData2(float[][][] data, Random rnd) {
		for(int i=0; i<data.length; i++){
			for(int j=0; j<data[i].length; j++){
				float pj = (float) (rnd.nextDouble()*200);
				float rad = (float) (pj*.3f+1);
				float ang = (float) (2*Math.PI*(i/3.0 + pj/(15.0*20)+Math.cos(rad*.03)));
				data[i][j][0] = (float) (Math.cos(ang)*rad+ rnd.nextGaussian()*6);
				data[i][j][1] = (float) (Math.sin(ang)*rad+ rnd.nextGaussian()*6);
			}
		}
	}

	ArrayList<ArrayList<float[]>> sampleLogs = new ArrayList<>();
	{
		for(@SuppressWarnings("unused") float[] s: sample)
			sampleLogs.add(new ArrayList<>());
	}

	//Choose a weight regularizer
	private Regularizer reg=new L2Reg(.000001);

	//Choose an activation function/nonlinearity
	//	private Neuron neuron=new Relu();
	private Neuron neuron=new Isrlu().fixA(4);

	BnType bnType;

	//Define the main network module as a stack of layers
	InOutModule mod;



	ArrayList<BatchNormoid> needsFinalize= new ArrayList<>(); 
	ArrayList<AbstractBatchNorm> needsInit = new ArrayList<>(); 

	private InOutModule bn() {
		switch(bnType) {
		case NONE: return new Copy();
		case BN: {
			BatchNorm bn = new BatchNorm();
			needsFinalize.add(bn);
			return bn;
		}
		case BRN: {
			BatchReNorm brn = new BatchReNorm();
			needsFinalize.add(brn);
			return brn;
		}
		case BLN: {
			BatchlessNorm bn = new BatchlessNorm(()->net.loss);
			needsInit.add(bn);
			return bn;
		}
		case BLNLOG: {
			BatchlessNormLog bn = new BatchlessNormLog(()->net.loss);
			needsInit.add(bn);
			return bn;
		}
		case BLNINV: {
			BatchlessNormInv bn = new BatchlessNormInv(()->net.loss);
			needsInit.add(bn);
			return bn;
		}
		default: return null;
		}
	}
	protected void finalizeBn() {
		TrainingBatchCursor cursor = cursor(data);

		for(int i=0; i<needsFinalize.size(); ++i) {
			BatchNormoid bn =needsFinalize.get(i);
			bn.setMode(FinalizationMode.MEANS, tr.getParams());
			eval.computeLoss(cursor, 1000);
			bn.setMode(FinalizationMode.STANDARD_DEVIATIONS, tr.getParams());
			eval.computeLoss(cursor, 1000);
			bn.setMode(FinalizationMode.NONE, tr.getParams());
		}
	}
	protected void initBn(){
		if(needsInit.isEmpty())
			return;
		ShuffledCursor cursor = new ShuffledCursor(cursor(data));
		cursor.shuffle();
		AbstractBatchNorm.setNormalizerParameters(eval, 128, cursor.subsequence(0, 128*2), needsInit);

	}

	//Another example of a network definition
	//	int k=3;
	//	int n=2;
	//	InOutModule mod2 = new StackModule()
	//			.addLayer(n, new Copy(0, null))
	//			.addLayer(Dropout.factory(0.7))
	//			.addLayer(3*n*k, new Dense().regularizer(reg))
	//			.addLayer(3*n, new Maxout())
	//			.addLayer(Dropout.factory(0.5))
	//			.addLayer(3*n*k, new Dense().regularizer(reg))
	//			.addLayer(3*n, new Maxout())
	//			.addLayer(Dropout.factory(0.5))
	//			.addLayer(3*n*k, new Dense().regularizer(reg))
	//			.addLayer(3*n, new Maxout())
	//			.addLayer(Dropout.factory(0.5))
	//			.addLayer(2*n*k, new Dense().regularizer(reg))
	//			.addLayer(2*n, new Maxout())
	//			.addFinalLayer(3, new Dense().regularizer(reg))
	//			;





	//buffer for the network input during test time
	float[] inputs=new float[2];
	//buffer for the network output during test time
	float[] outputs=new float[3];

	//The whole network, comprising main module and loss layer
	FeedForward net;



	private InOutModule makeStack() {
		boolean useBias = bnType==BnType.NONE;
		int firstLayerWidth = 50; //50
		int extraLayers = 20; //2
		
		int extraLayerWidth =40; //40
		StackModule ret = new StackModule();
		ret
				//			.addLayer(2, new BatchNorm())
				//Add a dense layer, followed by a batch normalization layer, with 50 channels each
				.addLayer(firstLayerWidth, new Dense(useBias).regularizer(reg), bn())
				//Add a nonlinearity layer using the chosen activation function
				.addLayer(firstLayerWidth, new Nonlin(neuron))
				//Add a dropout layer for better regularization
				.addLayer(Dropout.factory(0.9));
		for(int i=0; i<extraLayers; ++i) {
			ret
				//Add another dense and batch norm and nonlinearity layer, with 40 channels each
				.addLayer(extraLayerWidth, new Dense(useBias).regularizer(reg), bn(), new Nonlin(neuron))
				//Add a dropout layer for better regularization
				.addLayer(Dropout.factory(0.9));
		}
	
				//			//Add another dense and batch norm and nonlinearity layer, with 40 channels each
				//			.addLayer(40, new Dense().regularizer(reg), bn(), new Nonlin(neuron))
				//			//Add a dropout layer for better regularization
				//			.addLayer(Dropout.factory(0.9))
				//			//Add another dense and batch norm and nonlinearity layer, with 40 channels each
				//			.addLayer(40, new Dense().regularizer(reg), bn(), new Nonlin(neuron))
				//			//And another dropout layer
				//			.addLayer(Dropout.factory(0.9))
				//Add a final dense layer
				ret.addFinalLayer(3, new Dense().regularizer(reg));
		return ret;
	}
	private FeedForward makeNet() {
		return new FeedForward(mod, new SoftMaxNllLoss(), inputs.length, outputs.length);
	}


	//the data set to operate on
	float[][][] data;


	//The trainer for the network
	FeedForward.NaiveTrainer tr;
	//The evaluator for the network
	FeedForward.Eval eval;

	//For displaying the training data and the current networks classification
	JLabel disp;
	private BufferedImage dispOutput;

	//Whether to use minibatches, or just a single batch made of the whole data set.
	boolean miniBatches=true;

	//the batch size
	int batchSize;
	private float[][][] validation;
	int m = 15;
	double[] lastMLosses = new double[m];
	double[] sortedLosses = new double[m];
	double smoothedDiffLoss = 0;
	double smoothedDiffLossWeight = 0;
	double smoothedDiffLossDecay = 0.999;
	double latestOptimalLearnRate;
	boolean adaptLearnRate = true;
	public double latestOptimalLearnRate(){return latestOptimalLearnRate;}
	{
		Arrays.fill(lastMLosses, Double.POSITIVE_INFINITY);

	}



	public BatchlessTest2(float[][][] data, float[][][] validation, int batchSize, BnType normalizationType, boolean graphical){
		this.data=data;
		this.validation=validation;
		this.batchSize = batchSize;
		this.bnType = normalizationType;
		this.graphical=graphical;
		mod=makeStack();
		net=makeNet();
		//make the display
		JFrame jf=new JFrame();
		dispOutput=new BufferedImage(400, 400, BufferedImage.TYPE_INT_RGB);
		disp=new JLabel(new ImageIcon(dispOutput));
		jf.add(disp);
		jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		if(graphical) {
			jf.pack();
			jf.setVisible(true);
		}

		//determine batch size
		if(miniBatches) {
			//batchSize=128*8;
		}else {
			batchSize=0;
			for(float[][] a : data)
				batchSize+=a.length;
		}


		//Create the trainer and initialize the weights
		tr=net.getNaiveTrainer(batchSize, new AmsGrad());
//		tr=net.getNaiveTrainer(batchSize, new SGD());
		tr.initSmallWeights(1);
		optimalLearnRateSoFar=latestOptimalLearnRate=learnRate=adaptLearnRate?.0001:0.01;

		//create the evaluator
		eval=tr.getEvaluator(false);
		initBn();
	}

	boolean graphical;
	@Override
	public void run() {
		while(!done){
			batchTrain(miniBatches?1000000:100);
			if(graphical) {
				updateGraphics();
				disp.repaint();
			}
		}
		printResult();


	}



	private void printResult() {
		System.out.println("Batch size: "+batchSize);
		System.out.println("Normalization: "+bnType);
		System.out.println("Final validation loss: "+finalValidationLoss);
		System.out.println("Final training loss: "+finalTrainingLoss);
		System.out.println("Final learning rate: "+latestOptimalLearnRate);
		System.out.println("Final fluctuation: "+finalFluctuation);
		System.out.println("Batches processed unitl convergence: "+(batchesProcessed-1000));

	}	


	private void updateGraphics() {

		double txmin, txmax, tymin, tymax;
		txmin=tymin=Double.POSITIVE_INFINITY;
		txmax=tymax=Double.NEGATIVE_INFINITY;
		for(float[][] c: data)
			for(float[] i: c){
				txmax = Math.max(txmax, i[0]);
				tymax = Math.max(tymax, i[1]);
				txmin = Math.min(txmin, i[0]);
				tymin = Math.min(tymin, i[1]);
			}
		float margin = 0.3f;
		double minX = txmin - (txmax-txmin)*margin;
		double maxX = txmax + (txmax-txmin)*margin;
		double minY = tymin - (tymax-tymin)*margin;
		double maxY = tymax + (tymax-tymin)*margin;

		Color[] colors={
				Color.red, Color.green, Color.blue
		};


		int wx = dispOutput.getWidth();
		int wy = dispOutput.getHeight();
		Graphics g=dispOutput.getGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, wx, wy);
		long t0=System.currentTimeMillis();
		for(int ix=0; ix<wx; ix++){
			inputs[0]=(float) (ix*(maxX-minX)/wx+minX);
			for(int iy=0; iy<wy; iy++){
				inputs[1]=(float) (iy*(maxY-minY)/wy+minY);
				eval.eval(inputs, outputs);
				//				inst.forward(inputs, outputs);
				int mi=-1;
				double max=Float.NEGATIVE_INFINITY;
				for(int i=0; i<outputs.length; i++){
					if(outputs[i]>max){
						max=outputs[i];
						mi=i;
					}
				}
				if(mi != -1)
					dispOutput.setRGB(ix, iy, colors[mi].getRGB());
			}			
		}
		long t1=System.currentTimeMillis();
		System.out.println("Update time: "+(t1-t0));
		for(int c=0; c<data.length; c++){
			float[][] dc= data[c];
			for(int i=0; i<200 && i<dc.length; ++i){
				float[] dat = dc[i];
				int ix = (int) ((dat[0]-minX)*wx/(maxX-minX));
				int iy = (int) ((dat[1]-minY)*wy/(maxY-minY));
				int r = 4;
				g.setColor(colors[c]);
				g.fillOval(ix-r, iy-r, 2*r, 2*r);
				g.setColor(Color.BLACK);
				g.drawOval(ix-r, iy-r, 2*r, 2*r);
			}
		}
		for(int c=0; c<sample.length; c++){
			float[] dat = sample[c];
			int ix = (int) ((dat[0]-minX)*wx/(maxX-minX));
			int iy = (int) ((dat[1]-minY)*wy/(maxY-minY));
			int r = 2;
			g.drawOval(ix-r, iy-r, 2*r, 2*r);
		}
	}

	int batchesProcessed; 
	public double batchesTillConvergence() {return batchesProcessed-1000;}
	double minLoss = Double.POSITIVE_INFINITY;
	int lastLossDecrease;
	boolean startedSampleLogging;
	boolean done;
	double finalValidationLoss;
	public double finalValidationLoss() {return finalValidationLoss;}
	double finalTrainingLoss;
	public double finalTrainingLoss() {return finalTrainingLoss;}
	double finalFluctuation; 
	public double finalFluctuation() {return finalFluctuation;}

	double optimalLearnRateSoFar;
	double smallestSmoothedDiffLoss;
	int smoothedDiffLossIncreases;
	double oldMedianLoss;
	double oldMinimumLoss;
	double oldLoss;
	double learnRate;
	private void batchTrain(int its) {
		Random r = new Random();
		for(int it=0; it<its; it++){
			tr.learnRate = adaptLearnRate?learnRate*10:learnRate;
			if(miniBatches) {
				//randomly choose training examples
				for(int b=0; b<batchSize; ++b) {
					int c = r.nextInt(data.length);
					Arrays.fill(outputs, 0);
					outputs[c]=1;
					int i = r.nextInt(data[c].length);
					tr.addToBatch(data[c][i], outputs, 1);
				}
			}else {
				//put all data points into the batch
				for(int c=0; c<data.length; c++){
					Arrays.fill(outputs, 0);
					outputs[c]=1;
					for(int i=0; i<data[c].length; i++){
						tr.addToBatch(data[c][i], outputs, 1);
					}
				}
			}
			//End the batch and compute the training loss
			double loss=
					tr.endBatch();
			lastMLosses[batchesProcessed%m] = loss;
			System.arraycopy(lastMLosses, 0, sortedLosses, 0, m);
			Arrays.sort(sortedLosses);
			++batchesProcessed;

			double medianLoss = sortedLosses[m/2];
			double minimumLoss = sortedLosses[0];
			double diffLoss = batchesProcessed==1?0:loss-oldLoss;
//			double diffLoss = batchesProcessed<=sortedLosses.length?0:medianLoss - oldMedianLoss;
//			double diffLoss = batchesProcessed<=2?0:minimumLoss - oldMinimumLoss;
			//					+ (medianLoss - oldMedianLoss) * (1-smoothedDiffLossDecay);
			double newSmoothedDiffLoss = 
					smoothedDiffLoss * smoothedDiffLossDecay * smoothedDiffLossWeight 
					+ diffLoss * (1-smoothedDiffLossDecay);
			smoothedDiffLossWeight = smoothedDiffLossWeight * smoothedDiffLossDecay + (1-smoothedDiffLossDecay);
			smoothedDiffLoss = newSmoothedDiffLoss;
			oldMedianLoss = medianLoss;
			oldMinimumLoss = minimumLoss;
//			System.out.println(learnRate+" | " + loss + "| "+diffLoss+" | "+(smoothedDiffLoss/smoothedDiffLossWeight)+ " | "+smoothedDiffLossIncreases);;
			oldLoss = loss;

			int patience = 30;
			if(adaptLearnRate) {
				int sinceLearnRateReset = batchesProcessed%200;
				if(sinceLearnRateReset==0) {
					if(false && learnRate<1e-5)
						startedSampleLogging=true;
					//learnRate *= 0.5;
					
					optimalLearnRateSoFar=learnRate;
					smoothedDiffLossIncreases = 0;
					smallestSmoothedDiffLoss = smoothedDiffLoss/smoothedDiffLossWeight;
//					smoothedDiffLoss*=0.2;
//					smoothedDiffLossWeight*=0.2;
				}else if(sinceLearnRateReset>0) {
					if(smoothedDiffLossIncreases>=patience || sinceLearnRateReset == 1199) {
						learnRate = optimalLearnRateSoFar*0.7;
						latestOptimalLearnRate=learnRate;
						smoothedDiffLossIncreases=0;
						smallestSmoothedDiffLoss = smoothedDiffLoss/smoothedDiffLossWeight;

					}else {
						learnRate *= 1.01;
						if(smoothedDiffLoss/smoothedDiffLossWeight>smallestSmoothedDiffLoss)
							++smoothedDiffLossIncreases;
						else {
							smallestSmoothedDiffLoss = smoothedDiffLoss/smoothedDiffLossWeight;
							optimalLearnRateSoFar=learnRate;
							smoothedDiffLossIncreases=0;
						}
					}
				}
			}
			if(medianLoss<=minLoss) {
				minLoss = medianLoss;
				lastLossDecrease = batchesProcessed;
				//				System.out.println("Training loss decreased!");
			}else {
				if(batchesProcessed - lastLossDecrease>1000)
					startedSampleLogging=true;
			}
			if(startedSampleLogging) {
				logSamples();
				int logCount = sampleLogs.get(0).size();
				if(logCount==1000) {
					double radSum = 0;
					for(int i=0; i<sample.length; ++i)
						radSum+=informationRadius(sampleLogs.get(i), net.out.count);
					//					System.out.println("After logging for "+logCount+" batches:");
					double avgRad = radSum / sample.length;
					//					System.out.println("fluctuation = "+avgRad);
					done=true;
					finalFluctuation = avgRad;
					TrainingBatchCursor valCursor = cursor(validation);
					TrainingBatchCursor trCursor = cursor(data);
					finalizeBn();
					finalValidationLoss = eval.computeLoss(valCursor, 1000);
					finalTrainingLoss = eval.computeLoss(trCursor, 1000);
					break;

				}
			}
			//			System.out.println("loss = "+loss);
		}
	}
	private RandomAccess cursor(float[][][] dataset) {
		return TrainingBatchCursor.concat(
				cursor(dataset[0], 0),
				cursor(dataset[1], 1),
				cursor(dataset[2], 2)
				);
	}


	private RandomAccess cursor(float[][] fs, int i) {
		return new TrainingBatchCursor.RandomAccess() {
			int ni = fs[0].length;
			int no = 3;
			int at;
			@Override
			public TrainingBatchCursor.RandomAccess clone() {
				return cursor(fs, i);
			}
			@Override public double getWeight() {return 1;}
			@Override
			public void giveInputs(double[] inputs) {
				for(int i=0; i<ni; ++i)
					inputs[i] = fs[at][i];
			}
			@Override
			public void giveOutputs(double[] outputs) {
				Arrays.fill(outputs, 0);
				outputs[i]=1;
			}
			@Override public int inputCount() {return ni;}
			@Override public int outputCount() {return no;}
			@Override public boolean isFinished() {return at>=fs.length;}
			@Override public void next() {++at;}
			@Override public void reset() {at=0;}
			@Override public void seek(long position) {at=(int)position;}
			@Override public long size() {return fs.length;}

		};
	}



	private void logSamples() {
		for(int i=0; i<sample.length; ++i) {
			float[] samp = new float[net.out.count];
			eval.eval(sample[i], samp);
			float sum = 0;
			for(int j=0; j<samp.length; ++j)
				sum += samp[j] = (float)Math.exp(samp[j]);
			for(int j=0; j<samp.length; ++j)
				samp[j] /= sum;
			sampleLogs.get(i).add(samp);
		}
	}
	static HashSet<Thread> workers = new HashSet<>();
	static int activeWorkers;
	static ArrayDeque<Runnable> jobs = new ArrayDeque<>();
	static ArrayList<Runnable> runningJobs = new ArrayList<>();

	public void clear() {
		data = null;
		validation=null;
		mod=null;
		net=null;
		tr=null;
		eval=null;
		needsFinalize=null;
		needsInit=null;
		disp=null;
		dispOutput=null;
		lastMLosses = sortedLosses = null;
		sampleLogs=null;
	}
	public static void main(String[] args) throws InterruptedException {
		if(!true) {
			BatchlessTest2 inst = new BatchlessTest2(data2, data2Test, 64, BnType.BLN, !true);
			inst.run();
			return;
		}
		int pop = 100;
		int[] bss = {1, 2, 4, 8, 16, 32, 64	};
		BnType[] bnTypes = BnType.values();
		BatchlessTest2[][][] insts = new BatchlessTest2[bss.length][bnTypes.length][];
		for(int bsi = 0; bsi<bss.length; ++bsi) {
			int bs = bss[bsi];
			for(BnType bnType: bnTypes) {
				if(bs==1 && (bnType==BnType.BN || bnType==BnType.BRN))
					continue;
				BatchlessTest2[] batch = new BatchlessTest2[pop];
				for(int i=0; i<pop; ++i) {
					BatchlessTest2 inst = new BatchlessTest2(data2, data2Test, bs, bnType, false);
					submit(()->{inst.run(); inst.clear();});
					batch[i]=inst;
				}
				insts[bsi][bnType.ordinal()] = batch;
			}
		}
		Thread.sleep(1000);
		synchronized (workers) {
			while(!jobs.isEmpty() || activeWorkers!=0) {
				workers.wait(1000);
			}
			System.out.println();
		}
		printTable("Final validation losses", bss, insts, meanMetric(BatchlessTest2::finalValidationLoss), false);
		printTable("Final training losses", bss, insts, meanMetric(BatchlessTest2::finalTrainingLoss), false);
		printTable("Batches till convergence", bss, insts, rounded(meanMetric(BatchlessTest2::batchesTillConvergence)), true);
		printTable("Final fluctuation", bss, insts, meanMetric(BatchlessTest2::finalFluctuation), false);
		printTable("Final learn rate", bss, insts, meanMetric(BatchlessTest2::latestOptimalLearnRate), false);
		printTable("Final validation losses (spread)", bss, insts, sdevMetric(BatchlessTest2::finalValidationLoss), false);
		printTable("Final training losses (spread)", bss, insts, sdevMetric(BatchlessTest2::finalTrainingLoss), false);
		printTable("Batches till convergence (spread)", bss, insts, rounded(sdevMetric(BatchlessTest2::batchesTillConvergence)), true);
		printTable("Final fluctuation (spread)", bss, insts, sdevMetric(BatchlessTest2::finalFluctuation), false);
		printTable("Final learn rate (spread)", bss, insts, sdevMetric(BatchlessTest2::latestOptimalLearnRate), false);

	}
	public static void printTable(String title, int[] bss, BatchlessTest2[][][] data, 
			ToDoubleFunction<? super BatchlessTest2[]> metric, boolean integer) {
		System.out.println();
		System.out.println(title);
		BnType[] bnTypes = BnType.values();
		for(BnType bnType: bnTypes) {
			System.out.print(" & \t");
			System.out.print(bnType);
		}
		System.out.println("\\\\");
		System.out.println("\\hline");
		for(int bsi = 0; bsi<bss.length; ++bsi) {
			int bs = bss[bsi];
			System.out.print(bs);
			for(BnType bnType: bnTypes) {
				BatchlessTest2[] batch = data[bsi][bnType.ordinal()];
				System.out.print(" & \t");
				if(batch==null) {
					System.out.print(" -       ");
				}else {
					if(integer)
						System.out.print("$"+(int)(metric.applyAsDouble(batch))+"$");
					else
						System.out.print("$"+(float)(metric.applyAsDouble(batch))+"$");
				}
			}
			System.out.println("\\\\");
		}
	}
	public static <T> ToDoubleFunction<T> rounded(ToDoubleFunction<? super T> f){
		return x -> Math.round(f.applyAsDouble(x));
	}
	public static <T> ToDoubleFunction<T[]> meanMetric(ToDoubleFunction<? super T> extract){
		return batch -> {
			double sum = 0;
			int count = 0;
			for(T inst: batch) {
				double value = extract.applyAsDouble(inst);
				if(Double.isNaN(value))
					continue;
				sum += value;
				++count;
			}
			return sum/count;
		};
	}
	public static <T> ToDoubleFunction<T[]> sdevMetric(ToDoubleFunction<? super T> extract){
		return batch -> {
			double sum = 0;
			int count = 0;
			for(T inst: batch) {
				double value = extract.applyAsDouble(inst);
				if(Double.isNaN(value))
					continue;
				sum += value;
				++count;
			}
			double mean = sum/count;
			double var = 0;
			count = 0;
			for(T inst: batch) {
				double value = extract.applyAsDouble(inst);
				if(Double.isNaN(value))
					continue;
				double diff = value-mean;
				var += diff*diff;
				++count;
			}
			return Math.sqrt(var/(count-1));
		};
	}
	public static void submit(Runnable runThis) {
		synchronized (workers) {
			jobs.add(runThis);
			workers.notifyAll();			
			int par = Math.max(1, Runtime.getRuntime().availableProcessors());
			if(activeWorkers==workers.size() && workers.size()<par) {
				Thread worker = new Thread(()->{
					boolean active=false;
					try {
						int tries = 0;
						mainLoop:while(true) {
							Runnable job;
							synchronized(workers) {
								job=null;
								while(job==null) {
									job = jobs.poll();
									if(job!=null)
										break;
									++tries;
									if(tries>10)
										break mainLoop;
									if(active) {
										active=false;
										--activeWorkers;
										workers.notifyAll();

									}
									try {
										workers.wait(1000);
									} catch (InterruptedException e) {
										return;
									}

								}

								if(Thread.interrupted())
									return;
								if(!active) {
									++activeWorkers;
									active=true;
									workers.notifyAll();

								}
								runningJobs.add(job);

							}
							try {
								job.run();
								tries = 0;
							}finally {
								synchronized(workers) {
									runningJobs.remove(job);
									workers.notifyAll();
								}
							}
						}
					}finally {
						if(active) {
							active=false;
							synchronized(workers) {
								--activeWorkers;
								workers.notifyAll();
							}
						}
					}
				});
				worker.setDaemon(true);
				//				worker.setPriority(Thread.MIN_PRIORITY);
				worker.start();
				workers.add(worker);	
			}

		}
	}

	public static double informationRadius(ArrayList<float[]> p, int nj) {
		int ni = p.size();

		// Find distro q so that
		// sum_i sum_j p[i][j]*(log p[i][j] - log q[j]) 
		// is minimal, subject to condition sum_j q[j] = 1
		// The Lagrangian is sum_j ((sum_i  p[i][j]*(log p[i][j] - log q[j]))  + l*q[j]) - l 
		// differentiate w/rt l and all q[j] and set equal to 0
		// forall_j 0 = sum_i (p[i][j]/q[j]) + l
		//          0 = sum_j q[j] - 1
		// <=>
		// forall_j l = (avg_i p[i][j])/q[j]
		//          1 = sum_j q[j]
		// solution (because p[i] are normalized):
		// l = 1
		// q[j] = avg_i p[i][j]
		float[] q=new float[nj];
		for(float[] pi: p)
			for(int j=0; j<nj; ++j)
				q[j]+=pi[j];
		for(int j=0; j<nj; ++j)
			q[j] /= ni;
		double rad = 0;
		for(float[] pi: p)
			for(int j=0; j<nj; ++j) {
				float pij = pi[j];
				rad += pij==0?0:pij*(Math.log(pij/q[j]));
			}
		if(Double.isNaN(rad))
			System.out.println();
		return Math.max(0, rad/ni);
	}
}
