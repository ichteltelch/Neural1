package org.siquod.neural1.examples;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.siquod.neural1.Interface;
import org.siquod.neural1.Module;
import org.siquod.neural1.modules.BackpropStopper;
import org.siquod.neural1.modules.BatchNorm;
import org.siquod.neural1.modules.Copy;
import org.siquod.neural1.modules.Dense;
import org.siquod.neural1.modules.Dropout;
import org.siquod.neural1.modules.InOutCastFactory;
import org.siquod.neural1.modules.InOutCastLayer;
import org.siquod.neural1.modules.InOutModule;
import org.siquod.neural1.modules.Maxout;
import org.siquod.neural1.modules.Nonlin;
import org.siquod.neural1.modules.StackModule;
import org.siquod.neural1.modules.loss.SoftMaxNllLoss;
import org.siquod.neural1.modules.regularizer.L2Reg;
import org.siquod.neural1.modules.regularizer.Regularizer;
import org.siquod.neural1.net.FeedForward;
import org.siquod.neural1.neurons.Isrlu;
import org.siquod.neural1.neurons.Neuron;
import org.siquod.neural1.neurons.Relu;
import org.siquod.neural1.neurons.Tanh;


public class ClassifyTest {
	private Regularizer reg=new L2Reg(.000001);
//	private Neuron neuron=new Relu();
	private Neuron neuron=new Isrlu().fixA(4);
//	private Neuron neuron=new Tanh();

	InOutModule mod = new StackModule()
//			.addLayer(2, new BatchNorm())
			.addLayer(50, new Dense().regularizer(reg), new BatchNorm()
					)
	.addLayer(50, new Nonlin(neuron))
	.addLayer(Dropout.factory(0.9))
	.addLayer(40, new Dense().regularizer(reg), new BatchNorm(), 
			new Nonlin(neuron))
	.addLayer(Dropout.factory(.9))
//	.addLayer(20, new FullyConnected().regularizer(reg), new SimpleNonlinLayer(neuron))
	//.addFinalLayer(3, new FullyConnected(), new SimpleNonlinLayer(neuron))
	.addFinalLayer(3, new Dense().regularizer(reg))
	;
	int k=3;
	int n=2;
	InOutModule mod2 = new StackModule()
	.addLayer(n, new Copy(0, null))
	.addLayer(Dropout.factory(0.7))
	.addLayer(3*n*k, new Dense().regularizer(reg))
	.addLayer(3*n, new Maxout())
	.addLayer(Dropout.factory(0.5))
	.addLayer(3*n*k, new Dense().regularizer(reg))
	.addLayer(3*n, new Maxout())
	.addLayer(Dropout.factory(0.5))
	.addLayer(3*n*k, new Dense().regularizer(reg))
	.addLayer(3*n, new Maxout())
	.addLayer(Dropout.factory(0.5))
	.addLayer(2*n*k, new Dense().regularizer(reg))
	.addLayer(2*n, new Maxout())
	//.addFinalLayer(3, new FullyConnected(), new SimpleNonlinLayer(neuron))
	.addFinalLayer(3, new Dense().regularizer(reg))
	;

	static float[][][] data1;
	static float[][][] data2;
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
		data2=new float[3][200][2];
		for(int i=0; i<data2.length; i++){
			for(int j=0; j<data2[i].length; j++){
				float rad = j*.3f+1;
				float ang = (float) (2*Math.PI*(i/3.0 + j/(15.0*20)+Math.cos(rad*.03)));
				data2[i][j][0] = (float) (Math.cos(ang)*rad);
				data2[i][j][1] = (float) (Math.sin(ang)*rad);
			}
		}		
	}




	float[] inputs=new float[2];
	float[] outputs=new float[3];
	float[][][] data;
	//SequentialNet.SGDTrainer net;
	static float maf=.3f, mafi=1/maf;

	FeedForward net=new FeedForward(mod, new SoftMaxNllLoss(), inputs.length, outputs.length);
	FeedForward.NaiveTrainer tr;
	FeedForward.Eval eval;

	JLabel disp;

	private BufferedImage dispOutput;
	private ArrayList<float[]> allInputs=new ArrayList<>();



	public ClassifyTest(float[][][] data){
		this.data=data;
		for(float[][] dd: data)
			for(float[] d: dd)
				allInputs.add(d);
		JFrame jf=new JFrame();
		dispOutput=new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
		disp=new JLabel(new ImageIcon(dispOutput));
		jf.add(disp);
		jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jf.pack();
		jf.setVisible(true);

		int batchSize=0;
		for(float[][] a : data)
			batchSize+=a.length;
		tr=net.getNaiveTrainer(batchSize).initSmallWeights(1);
		tr.initSmallWeights(1);
		eval=tr.getEvaluator(false);

	}


	private void run() {
		while(true){
			batchTrain(1);
			//train(10000);
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
				Color.red, Color.green, Color.blue
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
		for(int it=0; it<its; it++){
			for(int c=0; c<data.length; c++){
				Arrays.fill(outputs, 0);
				outputs[c]=1;
				for(int i=0; i<data[c].length; i++){
					tr.addToBatch(data[c][i], outputs, 1);
				}
			}
			double loss=
					tr.endBatch();
			System.out.println(loss);
		}
	}
	public static void main(String[] args) {
		new ClassifyTest(data2).run();
	}
}
