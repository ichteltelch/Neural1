package org.siquod.neural1.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

import org.siquod.neural1.modules.BatchReNorm;
import org.siquod.neural1.modules.Copy;
import org.siquod.neural1.modules.Dense;
import org.siquod.neural1.modules.Dropout;
import org.siquod.neural1.modules.InOutModule;
import org.siquod.neural1.modules.LSTM_BN;
import org.siquod.neural1.modules.LayerNorm;
import org.siquod.neural1.modules.LogSoftmax;
import org.siquod.neural1.modules.Nonlin;
import org.siquod.neural1.modules.Para;
import org.siquod.neural1.modules.Sparse;
import org.siquod.neural1.modules.StackModule;
import org.siquod.neural1.modules.loss.NllLoss;
import org.siquod.neural1.modules.regularizer.L1Reg;
import org.siquod.neural1.modules.regularizer.Regularizer;
import org.siquod.neural1.net.Recurrent;
import org.siquod.neural1.net.Recurrent.Eval;
import org.siquod.neural1.net.Recurrent.NaiveTrainer;
import org.siquod.neural1.neurons.Neuron;
import org.siquod.neural1.neurons.SeLU;
import org.siquod.neural1.updaters.AmsGrad;

public class IirTextPrediction {
	/**
	 * Welche Zeichen sollen verwendet werden
	 */
	static String alphabet=
			"abcdefghijklmnopqrstuvwxyzüöäß"
			//			+ "ABCDEFGHIJKLMNOPQRSTUVWXYZAÖÜ"
			+ "<>/0123456789.:,;- \n?!\"%&="
			;
	/**
	 * Pfad zum Ordner mit den trainingsdaten
	 */
	static File corpusFolder=new File(new File(System.getProperty("user.home")), "progio/trainingdata/text");
	/**
	 * Zellen pro Schicht
	 */
	int width=512;
	/**
	 * Regularisierer
	 */
	private Regularizer reg=new L1Reg(.000001);
	/**
	 * Neuronentyp
	 */
	private Neuron neuron=new SeLU();
	/**
	 * Netzwerkstruktur
	 */
	InOutModule mod = new StackModule()
			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.9))
			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.9))
			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.9))
			.addLayer(alphabet.length(), output=new Dense().regularizer(reg))
			.addFinalLayer(new LogSoftmax())
			;

	
	static BitSet amask=new BitSet();
	ArrayList<int[]> corpus;
	boolean umb=true;
	private int batchSize=umb?5:1;
	private int subBatchSize=umb?1:5;
	int depth=1;
	private Dense output;
	int samplesSeen=0;
	//	private Neuron neuron=new Relu();
	//	private Neuron neuron=new Isrlu().fixA(10);
	//	private Neuron neuron=new Tanh();
	//	InOutModule mod = new StackModule()
	//			.shortcut()
	//			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.9))
	//			.addLayer(width, lstm(width, 2))
	//			.shortcut()
	//			.addLayer(Dropout.factory(0.8))
	//			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.8))
	//			.addLayer(width, lstm(width, 4))
	//			.shortcut()
	//			.addLayer(Dropout.factory(0.8))
	//			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.8))
	//			.addLayer(width, lstm(width, 2)).addLayer(Dropout.factory(0.8))
	//			.addLayer(width, lstm(width, 1)).addLayer(Dropout.factory(0.9))
	//			.shortcut()
	//			//			.addLayer(Dropout.factory(0.5))
	//			// 			.addLayer(512, new LSTM_BN(new FullyConnected(-1, null).regularizer(reg), new FullyConnected().regularizer(reg)))
	//			// 			.addLayer(512, new LSTM_BN(new FullyConnected(-1, null).regularizer(reg), new FullyConnected().regularizer(reg)))
	//			//			.addLayer(Dropout.factory(0.5))
	//			//						.addLayer(512, new LSTM(new FullyConnected(-1, null).regularizer(reg), new FullyConnected().regularizer(reg)))
	//			//			.addLayer(Dropout.factory(0.5))
	//			// 						.addLayer(4*alphabet.length(), new LSTM_BN(new FullyConnected(-1, null).regularizer(reg), new FullyConnected().regularizer(reg)))
	//			//			.addLayer(4*alphabet.length(), new FullyConnected().regularizer(reg), new BatchNorm(true), new SimpleNonlinLayer(new Tanh()), new SimpleNonlinLayer(new Tanh()))
	//			.addLayer(alphabet.length(), output=new Dense().regularizer(reg))
	//			.addFinalLayer(new LogSoftmax())
	//			;

	public InOutModule lstm(int width, int dt) {
		//		return lstm_sparse(0.1f, width, dt);
		return lstm_dense(width, dt);
	}
	public InOutModule lstm_sparse(float filling, int width, int dt) {
		return new Para(
				new LSTM_BN(new Sparse(filling, false, -dt, null).regularizer(reg), new Sparse(filling, false).regularizer(reg)),
				new Copy(),
				new StackModule().addFinalLayer(width, 
						new Sparse(filling, false),  
						new BatchReNorm(), new Nonlin(neuron)
						),
				new StackModule().addFinalLayer(width, 
						new Sparse(filling, false, -1, null),  
						new BatchReNorm(), new Nonlin(neuron)
						)
				);
	}
	public InOutModule lstm_dense(int width, int dt) {
		return new Para(
				new LSTM_BN(null, new Dense(false, -dt, null).regularizer(reg), new Dense(false).regularizer(reg)),
				new Copy(),
				new StackModule().addFinalLayer(width, 
						new Dense(false),  
						//						new BatchReNorm(), 
						new LayerNorm(), 
						new Nonlin(neuron)
						),
				new StackModule().addFinalLayer(width, 
						new Dense(false, -1, null),  
						//						new BatchReNorm(), 
						new LayerNorm(), 
						new Nonlin(neuron)
						)
				);
	}

	static{
		for(int i=0; i<alphabet.length(); ++i){
			amask.set(alphabet.charAt(i));
		}
	}
	float[][] inputs=new float[depth][alphabet.length()];
	float[][] outputs=new float[depth][alphabet.length()];
	AmsGrad u=new AmsGrad();
	Recurrent net=new Recurrent(mod, new NllLoss(), alphabet.length(), alphabet.length());
	NaiveTrainer tr=net.getNaiveTrainer(subBatchSize, depth).setUpdater(u).initSmallWeights(0.001);
	Eval eval=tr.getEvaluator(4, false);
	private int ig;

	public IirTextPrediction(ArrayList<int[]> load) {
		corpus=load;
		System.out.println(tr.getParams().size()+" Parameters");
		if(output != null) {
			int[] count=new int[alphabet.length()];
			int total=0;
			for(int[] is: corpus) {
				for(int i: is)
					++count[i];
				total+=is.length;
			}
			for(int i=0; i<count.length; ++i) {
				double b = Math.log(1+count[i])-Math.log(total);
				tr.getParams().set(output.getBias(), i, b);
			}
		}
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		IirTextPrediction app=new IirTextPrediction(load(corpusFolder, new ArrayList<int[]>()));
		app.run();
	}

	private void run() {
		tr.learnRate=0.001;
		int it=0;
		while(true){
			switch(it){
			case 0:
				if(umb)
					batchSize=5;
				else
					subBatchSize=5;
				tr.setBatchSizeAndLength(subBatchSize, depth=20);
				ig=3;
				tr.setUpdater(new AmsGrad());
				break;
			case 10:
				if(umb)
					batchSize=10;
				else
					subBatchSize=5;
				tr.setBatchSizeAndLength(subBatchSize, depth=30);
				ig=5;
				break;
			case 20:
				tr.setLength(depth=40);
				ig=7;
				break;
			case 30:
				if(umb)
					batchSize=30;
				else {
					subBatchSize=30;
					tr.setBatchSize(subBatchSize);
				}
				break;
			case 40:
				tr.setLength(depth=50);
				ig=8;
				break;
			case 50:
				tr.setLength(depth=70);
				ig=10;
				break;
			case 60:
				if(umb)
					batchSize=100;
				else {
					subBatchSize=100;
					tr.setBatchSize(subBatchSize);
				}
				break;
			case 70:
				tr.setLength(depth=100);
				break;
			}
			if(it==20 || it==100 || it%600==0)
				u.forgetVMax();

			if(depth!=inputs.length) {
				inputs=new float[depth][alphabet.length()];
				outputs=new float[depth][alphabet.length()];
			}

			System.out.println("training iteration: "+it);
			System.out.println("samples seen: "+samplesSeen);

			batchTrain(10, batchSize, subBatchSize);
			System.out.println(sample(1000));
			System.out.println();
			++it;
		}

	}	
	private String sample(int len) {
		StringBuilder ret=new StringBuilder();
		int[] file=corpus.get((int)(corpus.size()*Math.random()));
		int pos = (int) (Math.random()*(file.length));
		int lastChar=file[pos];
		double temp=1;
		eval.newRun();
		float[] inputs=this.inputs[0];
		float[] outputs=this.outputs[0];
		//		System.out.println(eval.showWeights());
		for(int i=0; i<len; ++i){
			Arrays.fill(inputs, 0);
			inputs[lastChar]=1;
			eval.step(inputs, outputs);
			double sum=0;
			for(int j=0; j<outputs.length; ++j)
				sum+=outputs[j]=(float) Math.exp(outputs[j]/temp);
			double rand=Math.random()*sum;
			int sample=-1;
			while(true){
				if(rand<0 || sample+1>=outputs.length)
					break;
				sample++;
				rand-=outputs[sample];
			}
			lastChar=sample;
			ret.append(alphabet.charAt(sample));
		}
		return ret.toString();

	}

//	private void gradCheck(int bs, long seed) {
//		Random rnd=new Random();
//		rnd.setSeed(seed);
//
//		fillBatch(bs, rnd);
//		double[] lossOut= {0};
//		ParamSet analytic=tr.gradientsOfBatch(lossOut);
//		ParamSet orig=tr.getParams();
//		ParamSet shiftDir=orig.clone();
//		for(int i=0; i<shiftDir.size(); ++i)
//			shiftDir.set(i, rnd.nextGaussian()*0.0001);
//		for(Module m: mod.deepSubmodules()) {
//			if(!(m instanceof BatchNorm))
//				continue;
//			BatchNorm b=(BatchNorm) m;
//			//			if(b.getAdd()!=null)
//			//				shiftDir.clear(b.getAdd());
//			//			shiftDir.clear(b.getMult());
//			shiftDir.clear(b.getRunningMean());
//			shiftDir.clear(b.getRunningVar());
//		}
//		ParamSet shifted=orig.clone();
//		shifted.addMultiple(shiftDir, 1);
//		rnd.setSeed(seed);
//		fillBatch(bs, rnd);
//		double shiftedLoss=tr.loss(shifted);
//		double analyticChange=analytic.dot(shiftDir);
//		double numericChange = shiftedLoss-lossOut[0];
//		System.out.println("Gradient check: "
//				+"\n  analytic:  "+analyticChange
//				+"\n  numerical: "+numericChange
//				);
//	}

	public void fillSubBatch(int bs, Random rnd) {
		for(int bi=0; bi<bs; bi++){
			int[] file=corpus.get(rnd.nextInt(corpus.size()));
			int pos = (int) (rnd.nextInt(file.length-depth-1));
			for(int d=0; d<depth; ++d) {
				Arrays.fill(inputs[d], 0);
				Arrays.fill(outputs[d], 0);
				inputs[d][file[pos+d]]=1;
				outputs[d][file[pos+d+1]]=1;
			}
			tr.addToBatch(inputs, outputs, 1, ig);
			//				avgLoss+=loss;
		}
	}

	private void batchTrain(int its, int bs, int sbs) {
		Random rnd=new Random();
		double avgLoss=0;
		for(int it=0; it<its; it++){
			//			gradCheck(bs, rnd.nextLong());
			double loss=0;
			for(int i=0; i<bs; ++i) {
				fillSubBatch(sbs, rnd);
				loss+=tr.endBatch();
				samplesSeen+=sbs;

			}
			loss/=bs;
			tr.endMetaBatch();
			System.out.println(loss);
			avgLoss+=loss;
		}			
		avgLoss/=its;
		System.out.println("Average loss: "+avgLoss);

		//		ArrayList<double[]> pop=new ArrayList<>();
		//		for(int i=0; i<10000; ++i) {
		//			int[] file=corpus.get((int)(corpus.size()*Math.random()));
		//			int pos = (int) (Math.random()*(file.length-depth-1));
		//			double[] inp=new double[inputs.length];
		//			for(int d=0; d<depth; ++d)
		//				inp[file[pos+d]+alphabet.length()*d]=1;
		//			pop.add(inp);
		//		}
		//		tr.determineStatistics(pop);
	}
	private static ArrayList<int[]> load(File f, ArrayList<int[]> ret) throws FileNotFoundException, IOException {
		if (f.getName().charAt(0)=='.')
			return ret;
		if(f.isDirectory()){
			for(File ff: f.listFiles())
				load(ff, ret);
			System.out.println("loaded folder "+f);
		}else{
			if(f.length()<200)
				return ret;
			try(FileReader fr=new FileReader(f);
					BufferedReader br=new BufferedReader(fr))
			{
				StringBuilder content=new StringBuilder();
				while(true){
					String line=br.readLine();
					if(line==null)
						break;
					if(line.equals("<!-- END WAYBACK TOOLBAR INSERT -->")){
						content=new StringBuilder();
						continue;
					}
					//					line=line.toLowerCase();
					line+='\n';
					for(int i=0; i<line.length(); ++i){
						char c = line.charAt(i);
						if(!amask.get(c))
							continue;
						content.append(c);
					}
				}
				int[] coded=new int[content.length()];
				for(int i=0; i<coded.length; ++i){
					coded[i]=alphabet.indexOf(content.charAt(i));
				}
				ret.add(coded);
			}
		}
		return ret;
	}
}
