package org.siquod.ml.neural1.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;

import org.siquod.ml.neural1.TensorFormat;
import org.siquod.ml.neural1.modules.BatchNorm;
import org.siquod.ml.neural1.modules.Conv1D;
import org.siquod.ml.neural1.modules.Copy;
import org.siquod.ml.neural1.modules.Dense;
import org.siquod.ml.neural1.modules.Dropout;
import org.siquod.ml.neural1.modules.InOutModule;
import org.siquod.ml.neural1.modules.LogSoftmax;
import org.siquod.ml.neural1.modules.MaxPooling1D;
import org.siquod.ml.neural1.modules.Maxout;
import org.siquod.ml.neural1.modules.Nonlin;
import org.siquod.ml.neural1.modules.StackModule;
import org.siquod.ml.neural1.modules.loss.NllLoss;
import org.siquod.ml.neural1.modules.regularizer.L2Reg;
import org.siquod.ml.neural1.modules.regularizer.Regularizer;
import org.siquod.ml.neural1.net.FeedForward;
import org.siquod.ml.neural1.neurons.Isrlu;
import org.siquod.ml.neural1.neurons.Neuron;

public class FirTextPrediction {
	
	static String alphabet="abcdefghijklmnopqrstuvwxyz<>/0123456789.:,;- \nüöäß?!\"";
	static BitSet amask=new BitSet();
	static File corpusFolder=new File(new File(System.getProperty("user.home")), "progio/trainingdata/text");
	ArrayList<int[]> corpus;
	int depth=9;
	private int batchSize=5000;
	private Regularizer reg=new L2Reg(.001);
//	private Neuron neuron=new Relu();
	private Neuron neuron=new Isrlu().fixA(10);
//	private Neuron neuron=new Tanh();
	int dd=depth;
	int channels = 6*alphabet.length();
	InOutModule mod = new StackModule()
//			.addLayer(new TensorFormat(dd, alphabet.length()), new Conv1D(0).regularizer(reg))
//		.addLayer(new TensorFormat(dd, 2*alphabet.length()), new FullyConnected().regularizer(reg))
			.addLayer(new TensorFormat(dd, channels),
					new Conv1D(1).regularizer(reg),  new BatchNorm(), 
					new Nonlin(neuron))
			.addLayer(new TensorFormat(dd, channels),
					new Conv1D(1).regularizer(reg),  new BatchNorm(), 
					new Nonlin(neuron))
			.addLayer(new TensorFormat(dd=(dd+1)/2, channels), new MaxPooling1D(),
					new Conv1D(1).regularizer(reg),  new BatchNorm(), 
					new Nonlin(neuron))
			.addLayer(new TensorFormat(dd=(dd+1)/2, channels), new MaxPooling1D(),
					new Conv1D(1).regularizer(reg),  new BatchNorm(), 
					new Nonlin(neuron))

//			.addLayer(2*alphabet.length(), new FullyConnected().regularizer(reg), new SimpleNonlinLayer(neuron), new BatchNorm())
			.addLayer(alphabet.length(), new Dense().regularizer(reg))
			.addFinalLayer(new LogSoftmax());

			;

	InOutModule mod1 = new StackModule()
//	.addLayer(depth*alphabet.length(), new CopyLayer(0, null))
//	.addLayer(Dropout.factory(0.9))
	
	.addLayer(2*depth*alphabet.length(), new Dense().regularizer(reg), new Nonlin(neuron))
	.addLayer(Dropout.factory(0.9, false))
	
//	.shortcut(alphabet.length()*depth/2)
	.addLayer(depth*alphabet.length(), new Dense().regularizer(reg), new Nonlin(neuron))
	.addLayer(Dropout.factory(0.9, false))

//	.addLayer(depth/3*alphabet.length(), new FullyConnected().regularizer(reg), new SimpleNonlinLayer(neuron))
//	.addLayer(Dropout.factory(0.9))
//	.shortcut(alphabet.length())
	
	.addLayer(alphabet.length(), new Dense().regularizer(reg))
	.endShortcut()
	.addFinalLayer(new LogSoftmax());
	;
	int k=3;
	InOutModule mod2 = new StackModule()
	.addLayer(depth*alphabet.length(), new Copy(0, null))
	.addLayer(Dropout.factory(0.9, false))
	.addLayer(depth*alphabet.length()*k, new Dense().regularizer(reg))
	.addLayer(depth*alphabet.length(), new Maxout())
	.addLayer(Dropout.factory(0.9, false))
	.addLayer(depth*alphabet.length()*k, new Dense().regularizer(reg))
	.addLayer(depth*alphabet.length(), new Maxout())
	.addLayer(Dropout.factory(0.9, false))
	.addLayer(depth/2*alphabet.length()*k, new Dense().regularizer(reg))
	.addLayer(depth/2*alphabet.length(), new Maxout())
//	.addLayer(Dropout.factory(0.5))
	
//	.shortcut(alphabet.length()*depth/2)
//	.addLayer(depth*alphabet.length()*k, new FullyConnected().regularizer(reg))
//	.addLayer(depth*alphabet.length(), new Maxout())
//	.addLayer(Dropout.factory(0.7))
//	.shortcut(alphabet.length())
	
	.addLayer(alphabet.length(), new Dense().regularizer(reg))
	.endShortcut()
	.addFinalLayer(new LogSoftmax());
	;
	static{
		for(int i=0; i<alphabet.length(); ++i){
			amask.set(alphabet.charAt(i));
		}
	}
	float[] inputs=new float[alphabet.length()*depth];
	float[] outputs=new float[alphabet.length()];
//	FeedForward net=new FeedForward(mod, new NllLoss(), inputs.length, outputs.length);
	FeedForward net=new FeedForward(mod, new NllLoss(), new TensorFormat(depth, alphabet.length()), 
			new TensorFormat(alphabet.length())).init();

	FeedForward.NaiveTrainer tr=net.getNaiveTrainer(batchSize, null).initSmallWeights(0.1);
	FeedForward.Eval eval=tr.getEvaluator(false);

	public FirTextPrediction(ArrayList<int[]> load) {
		corpus=load;
		System.out.println(tr.getParams().size()+" Parameters");
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		FirTextPrediction app=new FirTextPrediction(load(corpusFolder, new ArrayList<int[]>()));
		app.run();
	}

	private void run() {
		tr.learnRate=0.01;
		while(true){
			batchTrain(10, batchSize);
			System.out.println(sample(1000));
			System.out.println();		
		}

	}	
	private String sample(int len) {
		StringBuilder ret=new StringBuilder();
		int[] file=corpus.get((int)(corpus.size()*Math.random()));
		int pos = (int) (Math.random()*(file.length-depth-1));
		int[] buffer=new int[depth];
		for(int d=0; d<depth; ++d)
			buffer[d]=file[pos+d];
		double temp=1;
//		System.out.println(eval.showWeights());
		for(int i=0; i<len; ++i){
			Arrays.fill(inputs, 0);
			for(int d=0; d<depth; ++d)
				inputs[buffer[d]+d*alphabet.length()]=1;
			eval.eval(inputs, outputs);
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
			for(int d=0; d<depth-1; ++d)
				buffer[d]=buffer[d+1];
			buffer[buffer.length-1]=sample;
			
			ret.append(alphabet.charAt(sample));
			
		}
		return ret.toString();
		
	}

	private void batchTrain(int its, int bs) {
		for(int it=0; it<its; it++){
			double avgLoss=0;
			for(int bi=0; bi<bs; bi++){
				int[] file=corpus.get((int)(corpus.size()*Math.random()));
				int pos = (int) (Math.random()*(file.length-depth-1));
				Arrays.fill(inputs, 0);
				Arrays.fill(outputs, 0);
				for(int d=0; d<depth; ++d)
					inputs[file[pos+d]+alphabet.length()*d]=1;
				outputs[file[pos+depth]]=1;
				tr.addToBatch(inputs, outputs, 1);
//				avgLoss+=loss;
			}
			avgLoss=tr.endBatch();
			System.out.println(avgLoss);
		}

	}
	private static ArrayList<int[]> load(File f, ArrayList<int[]> ret) throws FileNotFoundException, IOException {
		if(f.isDirectory()){
			for(File ff: f.listFiles())
				load(ff, ret);
			System.out.println("loaded folder "+f);
		}else{
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
					line=line.toLowerCase()+'\n';
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
