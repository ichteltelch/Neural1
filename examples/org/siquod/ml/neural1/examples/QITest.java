package org.siquod.ml.neural1.examples;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Random;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.modules.BatchReNorm;
import org.siquod.ml.neural1.modules.Copy;
import org.siquod.ml.neural1.modules.Dense;
import org.siquod.ml.neural1.modules.Dropout;
import org.siquod.ml.neural1.modules.InOutCastFactory;
import org.siquod.ml.neural1.modules.InOutCastLayer;
import org.siquod.ml.neural1.modules.InOutModule;
import org.siquod.ml.neural1.modules.Nonlin;
import org.siquod.ml.neural1.modules.ParameterizedNonlin;
import org.siquod.ml.neural1.modules.QuadraticInteraction;
import org.siquod.ml.neural1.modules.StackModule;
import org.siquod.ml.neural1.modules.loss.SoftMaxNllLoss;
import org.siquod.ml.neural1.modules.regularizer.L2Reg;
import org.siquod.ml.neural1.modules.regularizer.Regularizer;
import org.siquod.ml.neural1.net.FeedForward;
import org.siquod.ml.neural1.neurons.FadeInNeuron;
import org.siquod.ml.neural1.neurons.Isrlu;
import org.siquod.ml.neural1.neurons.Neuron;
import org.siquod.ml.neural1.optimizers.*;


public class QITest {

	//Example data sets
//	static float[][][] data1;
//	static float[][][] data2;
	static float[][][] data3;
	static{
		Random rnd=new Random(12345);
//		data1=new float[3][100][2];
//		float[] cx = {1,-.5f,-.5f};
//		float[] cy = {0,(float) Math.sqrt(.75),(float) -Math.sqrt(.75)};
//		for(int i=0; i<data1.length; i++){
//			for(int j=0; j<data1[i].length; j++){
//				data1[i][j][0] = (float) (cx[i] + rnd.nextGaussian()*.5);
//				data1[i][j][1] = (float) (cy[i] + rnd.nextGaussian()*.5);
//			}
//		}
//		data2=new float[3][200][2];
//		for(int i=0; i<data2.length; i++){
//			for(int j=0; j<data2[i].length; j++){
//				float rad = j*.3f+1;
//				float ang = (float) (2*Math.PI*(i/3.0 + j/(15.0*20)+Math.cos(rad*.03)));
//				data2[i][j][0] = (float) (Math.cos(ang)*rad+ rnd.nextGaussian()*6);
//				data2[i][j][1] = (float) (Math.sin(ang)*rad+ rnd.nextGaussian()*6);
//			}
//		}		
		data3=new float[3][200][2];
		for(int i=0; i<data3.length; i++){
			for(int j=0; j<data3[i].length; j++){
				float rad = (float) (i*20);
				float ang = (float) (2*Math.PI*rnd.nextDouble());
				data3[i][j][0] = (float) (Math.cos(ang)*rad+ rnd.nextGaussian()*8);
				data3[i][j][1] = (float) (Math.sin(ang)*rad+ rnd.nextGaussian()*8);
			}
		}		
	}
	
	
	
	//Choose a weight regularizer
	private Regularizer reg=new L2Reg(.000001);

	//Choose an activation function/nonlinearity
	//	private Neuron neuron=new Relu();
	private Neuron neuron=new Isrlu().fixA(4);

	//Define the main network module as a stack of layers
	InOutModule mod = new StackModule()
			//Add a final dense layer
			.addLayer(5, QuadraticInteraction.b()
//					.repetitions(3).kernel(1, 1, 1)
					.repetitions(2).symmetricKernel(2, 1)
//					.bothModules((i, o)->new Copy())
					//.outputModule((i, o)->new Dense())
					.build()
					)
			.addLayer(5, new ParameterizedNonlin(new FadeInNeuron(neuron)))
			.addLayer(2, new Dense().regularizer(reg))
			.addFinalLayer(2, new Copy())
			;
	
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
	FeedForward net=new FeedForward(mod, new SoftMaxNllLoss(), inputs.length, outputs.length).init();

	
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
	boolean miniBatches=false;

	//the batch size
	int batchSize;



	public QITest(float[][][] data){
		this.data=data;
		
		//make the display
		JFrame jf=new JFrame();
		dispOutput=new BufferedImage(400, 400, BufferedImage.TYPE_INT_RGB);
		disp=new JLabel(new ImageIcon(dispOutput));
		jf.add(disp);
		jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jf.pack();
		jf.setVisible(true);

		//determine batch size
		if(miniBatches) {
			batchSize=128;
		}else {
			batchSize=0;
			for(float[][] a : data)
				batchSize+=a.length;
		}
		

		//Create the trainer and initialize the weights
		tr=net.getNaiveTrainer(batchSize, new AmsGrad());
		tr.initSmallWeights(1);
		tr.learnRate=0.01;

		//create the evaluator
		eval=tr.getEvaluator(false);

	}


	private void run() {
		while(true){
			batchTrain(miniBatches?1:100);
			updateGraphics();
			disp.repaint();
		}

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
				new Color(1, 0, 0, 0.7f),
				new Color(0, 1, 0, 0.7f),
				new Color(0, 0, 1, 0.7f),
		};


		int wx = dispOutput.getWidth();
		int wy = dispOutput.getHeight();
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
		Graphics g=dispOutput.getGraphics();
		for(int c=0; c<data.length; c++){
			for(float[] dat: data[c]){
				int ix = (int) ((dat[0]-minX)*wx/(maxX-minX));
				int iy = (int) ((dat[1]-minY)*wy/(maxY-minY));
				int r = 4;
				g.setColor(colors[c]);
				g.fillOval(ix-r, iy-r, 2*r, 2*r);
				g.setColor(Color.BLACK);
				g.drawOval(ix-r, iy-r, 2*r, 2*r);
			}
		}

	}

	private void batchTrain(int its) {
		Random r = new Random();
		for(int it=0; it<its; it++){
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
			//End the batch and comput the training loss
			double loss=
					tr.endBatch();
			System.out.println(loss);
		}
	}
	public static void main(String[] args) {
		new QITest(data3).run();
	}
}
