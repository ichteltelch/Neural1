package org.siquod.ml.neural1.examples;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import org.siquod.ml.data.TrainingBatchCursor;
import org.siquod.ml.data.Whitener;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.modules.AbstractBatchNorm;
import org.siquod.ml.neural1.modules.BatchNorm;
import org.siquod.ml.neural1.modules.BatchNormoid;
import org.siquod.ml.neural1.modules.BatchNormoid.FinalizationMode;
import org.siquod.ml.neural1.modules.BatchReNorm;
import org.siquod.ml.neural1.modules.BatchlessNormLog;
import org.siquod.ml.neural1.modules.BatchlessNormInv;
import org.siquod.ml.neural1.modules.Copy;
import org.siquod.ml.neural1.modules.Dense;
import org.siquod.ml.neural1.modules.InOutModule;
import org.siquod.ml.neural1.modules.Nonlin;
import org.siquod.ml.neural1.modules.StackModule;
import org.siquod.ml.neural1.modules.loss.L2Loss;
import org.siquod.ml.neural1.net.FeedForward;
import org.siquod.ml.neural1.neurons.Neuron;
import org.siquod.ml.neural1.neurons.LeakyRelu;
import org.siquod.ml.neural1.optimizers.Adam;
import org.siquod.ml.neural1.optimizers.AmsGrad;
import org.siquod.ml.neural1.optimizers.Rprop;
import org.siquod.ml.neural1.modules.regularizer.Regularizer;
import org.siquod.ml.neural1.modules.regularizer.L2UnitReg;

public class BatchlessTest3 implements Runnable{
	static enum BnType{
		NONE,
		BN,
		BRN,
		BLNLOG,
		BLNINV,
	}
	//Example data sets

	//Choose an activation function/nonlinearity
	//	private Neuron neuron=new Relu();
	private Neuron neuron=new LeakyRelu().fixA(1f);

	Regularizer reg = new L2UnitReg(0.00001);

	BnType bnType;

	//Define the main network module as a stack of layers
	InOutModule mod;

	static int layers = 3;

	private InOutModule makeStack() {
		StackModule mod = new StackModule();
		//		mod.addLayer(1, bn());
		for(int i=0; i<layers; ++i) {
			Dense d = new Dense();
			denses.add(d);
			mod.addLayer(2, d, bn(), new Nonlin(neuron));
		}
		mod.addFinalLayer(1, new Dense());

		return mod;
	}
	long scrambleSeed = 123;
	public void scramble() {
		Random ra = new Random(scrambleSeed);
		ParamSet params = tr.getParams();
		for(Dense d: denses) {
			double scale = Math.exp(ra.nextDouble()*10-8);
			for(int i=0; i<d.weights.count; ++i)
				params.set(d.weights, i, scale * (ra.nextDouble()-0.5));
			for(int i=0; i<d.bias.count; ++i)
				params.set(d.bias, i, ra.nextDouble()*100-50);
		}
	}

	ArrayList<Dense> denses = new ArrayList<>(); 
	ArrayList<BatchNormoid> bnoids = new ArrayList<>(); 
	ArrayList<AbstractBatchNorm> abns = new ArrayList<>(); 

	private InOutModule bn() {
		switch(bnType) {
		case NONE: return new Copy();
		case BN: 
			BatchNorm bn = new BatchNorm();
			bnoids.add(bn);
			abns.add(bn);
			return bn;
		case BRN: 
			BatchReNorm brn = new BatchReNorm();
			bnoids.add(brn);
			abns.add(brn);
			return brn;
		case BLNLOG: 
			BatchlessNormLog bln = new BatchlessNormLog(()->net.loss);
			abns.add(bln);
			return bln;
		case BLNINV: 
			BatchlessNormInv bln2 = new BatchlessNormInv(()->net.loss);
			abns.add(bln2);
			return bln2;
		default: return null;
		}
	}
	protected void finalize(int count) {
		TrainingBatchCursor cursor = cursor(123, count);

		for(int i=0; i<bnoids.size(); ++i) {
			BatchNormoid bn =bnoids.get(i);
			bn.setMode(FinalizationMode.MEANS, tr.getParams());
			eval.computeLoss(cursor, 1000);
			bn.setMode(FinalizationMode.STANDARD_DEVIATIONS, tr.getParams());
			eval.computeLoss(cursor, 1000);
			bn.setMode(FinalizationMode.NONE, tr.getParams());
		}
	}
	double[] computeErrorsPerLayer(int sampleSize) {
		ParamSet ps = tr.getParams();
		TrainingBatchCursor cursor = cursor(1235, sampleSize);
		//Whitener w = Whitener.gaussianForInputs(cursor);
		double[] errors = new double[abns.size()+1];
		//		AbstractBatchNorm l0 = abns.get(0);
		//		System.out.println("sigma: "+l0.mapSigmaFromStorage(ps.get(l0.sigmaSotrage(), 0)));
		//		System.out.println("mean:  "+ps.get(l0.meanStorage(), 0));
		double loss = eval.computeLoss(cursor, 1000, false, ass->{
			for(int i=0; i<errors.length-1; ++i) {
				errors[i+1] = abns.get(i).distributionMismatch(ass, ps, 0);
			}
		});
		errors[0]=loss;
		for(int i=1; i<errors.length; ++i) {
			errors[i]/=sampleSize;
		}
		return errors;
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
	float[] inputs=new float[1];
	//buffer for the network output during test time
	float[] outputs=new float[1];

	//The whole network, comprising main module and loss layer
	FeedForward net;



	private FeedForward makeNet() {
		return new FeedForward(mod, new L2Loss(), inputs.length, outputs.length);
	}




	//The trainer for the network
	FeedForward.NaiveTrainer tr;
	//The evaluator for the network
	FeedForward.Eval eval;



	//the batch size
	int batchSize;



	public BatchlessTest3(int batchSize, BnType normalizationType, boolean graphical){
		this.batchSize = batchSize;
		this.bnType = normalizationType;
		mod=makeStack();
		net=makeNet();
		//make the display




		//Create the trainer and initialize the weights
		tr=net.getNaiveTrainer(batchSize, new Rprop());
		tr.initSmallWeights(1);
		tr.learnRate=0.0001;

		//create the evaluator
		eval=tr.getEvaluator(false);

	}

	@Override
	public void run() {
		scramble();
		while(!done){
			//			finalize(10000);
			double[] errors = computeErrorsPerLayer(10000);
			System.out.println("After batch "+batchesProcessed+": "+Arrays.toString(errors));
			errorLog.add(errors);
			if(batchesProcessed>300)
				done=true;
			batchTrain(5);
		}
		printResult();


	}



	private void printResult() {
		System.out.println("Batch size: "+batchSize);
		System.out.println("Normalization: "+bnType);
	}	


	int batchesProcessed;
	boolean done;
	Random rnd = new Random();
	ArrayList<double[]>  errorLog = new ArrayList<>();
	private void batchTrain(int its) {
		for(int it=0; it<its; it++){
			//randomly choose training examples
			for(int b=0; b<batchSize; ++b) {
				float sample = (float)rnd.nextGaussian();
				outputs[0]=sample;
				inputs[0]=sample;
				tr.addToBatch(inputs, outputs, 1);
			}
			tr.endBatch();

			++batchesProcessed;

			//			System.out.println("loss = "+loss);
		}
	}



	private TrainingBatchCursor cursor(long seed, int n) {
		return new TrainingBatchCursor() {
			int at;
			Random rnd = new Random(seed);
			double sample = rnd.nextGaussian();
			@Override
			public TrainingBatchCursor clone() {
				return cursor(seed, n);
			}
			@Override public double getWeight() {return 1;}
			@Override
			public void giveInputs(double[] inputs) {
				inputs[0] = sample;
			}
			@Override
			public void giveOutputs(double[] outputs) {
				outputs[0] = sample;
			}
			@Override public int inputCount() {return 1;}
			@Override public int outputCount() {return 1;}
			@Override public boolean isFinished() {return at>=n;}
			@Override public void next() {
				++at; 				
				sample = rnd.nextGaussian();
			}	
			@Override
			public void reset() {
				at = 0;
				rnd.setSeed(seed);
				sample = rnd.nextGaussian();
			}

		};
	}




	static HashSet<Thread> workers = new HashSet<>();
	static int activeWorkers;
	static ArrayDeque<Runnable> jobs = new ArrayDeque<>();
	static ArrayList<Runnable> runningJobs = new ArrayList<>();

	public static void main(String[] args) throws InterruptedException, IOException {
		//		if(true) {
		//			BatchlessTest3 inst = new BatchlessTest3(64, BnType.BLNLOG, true);
		//			inst.run();
		//			return;
		//		}

		int[] bss = {1, 2, 4, 8, 16, 32, 64};
		BnType[] bnTypes = BnType.values();
		int scrambles = 10;
		BatchlessTest3[][][] insts = new BatchlessTest3[bss.length][bnTypes.length][scrambles];
		for(int bsi = 0; bsi<bss.length; ++bsi) {
			int bs = bss[bsi];
			for(BnType bnType: bnTypes) {
				if(bs==1 && (bnType==BnType.BN || bnType==BnType.BRN))
					continue;
				for(int i=0; i<scrambles; ++i) {
					BatchlessTest3 inst = new BatchlessTest3(bs, bnType, false);
					inst.scrambleSeed = 123+i;
					submit(inst);
					insts[bsi][bnType.ordinal()][i] = inst;
				}
			}
		}
		Thread.sleep(1000);
		synchronized (workers) {
			while(!jobs.isEmpty() || activeWorkers!=0) {
				workers.wait(1000);
			}
			System.out.println();
		}
		File folder = new File("/home/bb/DesktopFolderViews/Schreiben/Fulgamentis.dir/Papers/BatchlessNormalization/data");
		folder.mkdirs();
		for(int bsi = 0; bsi<bss.length; ++bsi) {
			int bs=bss[bsi];
			for(int i=0; i<=layers; ++i) {
				String id = i==0?"loss":"layer_"+(i);
				File file = new File(folder, id+"-bs_"+bs+".csv");
				ArrayList<BatchlessTest3[]> cols = new ArrayList<>();
				int rowCount = Integer.MAX_VALUE;
				bnType:for(BnType bnType: bnTypes) {
					BatchlessTest3[] instarr = insts[bsi][bnType.ordinal()];
					if(instarr==null)
						continue;
					for(BatchlessTest3 inst: instarr) {
						if(inst==null || inst.errorLog.get(0).length<=i)
							continue bnType;
					}
					cols.add(instarr);
					for(BatchlessTest3 inst: instarr)
						rowCount = Math.min(rowCount, inst.errorLog.size());
				}
				if(cols.isEmpty())
					return;
				try(FileWriter fw = new FileWriter(file);
						BufferedWriter bw = new BufferedWriter(fw)){
					for(BatchlessTest3[] inst: cols) {
						bw.write(inst[0].bnType.toString());
						bw.write(",");
					}
					bw.write("\n");
					for(int row = 0; row < rowCount; ++row) {
						for(BatchlessTest3[] instarr: cols) {
							double avg = 0;
							for(BatchlessTest3 inst: instarr)
								avg += inst.errorLog.get(row)[i];
							bw.write(String.valueOf(avg/instarr.length));
							bw.write(",");
						}
						bw.write("\n");

					}


				}
			}
		}




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
		return rad/ni;
	}
}
