package org.siquod.neural1.data;

import java.util.function.Consumer;

public interface Whitener {
	void whiten(double[] in, double[] out);
	int dim();
	static class GaussianWhitener implements Whitener{
		public final double[] µ;
		public final double[] invSigma;
		public GaussianWhitener(double[] µ, double[] invSigma) {
			this.µ=µ;
			this.invSigma=invSigma;
		}
		@Override
		public void whiten(double[] in, double[] out) {
			for(int i=0; i<µ.length; ++i) {
				out[i] = (in[i]-µ[i])*invSigma[i];
			}
			
		}
		@Override
		public int dim() {
			return µ.length;
		}
		public void colorize(double[] inWhite, double[] outColor) {
			for(int i=0; i<µ.length; ++i) {
				outColor[i] = inWhite[i]/invSigma[i]+µ[i];
			}
		}
		public void colorizeLogarithmOfSigma(double[] inWhite, double[] outColor) {
			for(int i=0; i<µ.length; ++i) {
				outColor[i] = inWhite[i]-Math.log(invSigma[i]);
			}
		}
	}
	public static GaussianWhitener gaussianForInputs(TrainingBatchCursor data) {
		int dim = data.inputCount();
		Consumer<double[]> give = a->data.giveInputs(a);
		return makeGaussianWhitener(data, dim, give);
	}
	public static GaussianWhitener gaussianForOutputs(TrainingBatchCursor data) {
		int dim = data.outputCount();
		Consumer<double[]> give = a->data.giveOutputs(a);
		return makeGaussianWhitener(data, dim, give);
	}
	public static GaussianWhitener gaussianForInputs(TrainingBatchCursor data, int[]... sigmaGroups) {
		int dim = data.inputCount();
		Consumer<double[]> give = a->data.giveInputs(a);
		return makeGaussianWhitener(data, dim, give, sigmaGroups);
	}
	public static GaussianWhitener gaussianForOutputs(TrainingBatchCursor data, int[]... sigmaGroups) {
		int dim = data.outputCount();
		Consumer<double[]> give = a->data.giveOutputs(a);
		return makeGaussianWhitener(data, dim, give, sigmaGroups);
	}
	static GaussianWhitener makeGaussianWhitener(TrainingBatchCursor data, int dim, Consumer<double[]> give) {
		double[] µ=new double[dim];
		double[] buffer=new double[dim];
		double n=0;
		data.reset();
		boolean all1=true;
		while(!data.isFinished()) {
			double w = data.getWeight();
			all1 &= w==1;
			n+=w;
			give.accept(buffer);
			for(int i=0; i<dim; ++i)
				µ[i]+=w*buffer[i];
			data.next();
		}
		double µNorm=1/n;
		for(int i=0; i<dim; ++i)
			µ[i]*=µNorm;
		double[] sigma = new double[dim];
		data.reset();
		while(!data.isFinished()) {
			give.accept(buffer);
			for(int i=0; i<dim; ++i) {
				double diff=buffer[i]-µ[i];
				sigma[i]+=diff*diff*data.getWeight();
			}
			data.next();
		}
		double isNorm=(all1?n-1:n);
		for(int i=0; i<dim; ++i)
			sigma[i]=Math.sqrt(isNorm/sigma[i]);
		return new GaussianWhitener(µ, sigma);
	}
	static GaussianWhitener makeGaussianWhitener(TrainingBatchCursor data, int dim, Consumer<double[]> give, int[]... sigmaGroups) {
		double[] µ=new double[dim];
		double[] buffer=new double[dim];
		double n=0;
		data.reset();
		boolean all1=true;
		while(!data.isFinished()) {
			double w = data.getWeight();
			all1 &= w==1;
			n+=w;
			give.accept(buffer);
			for(int i=0; i<dim; ++i)
				µ[i]+=w*buffer[i];
			data.next();
		}
		double µNorm=1/n;
		for(int i=0; i<dim; ++i)
			µ[i]*=µNorm;
		double[] sigma = new double[sigmaGroups.length];
		double[] invSigma = new double[dim];
		data.reset();
		while(!data.isFinished()) {
			give.accept(buffer);
			for(int sgi = 0; sgi<sigmaGroups.length; ++sgi) {
				int[] sg = sigmaGroups[sgi];
				for(int i: sg) {
					double diff=buffer[i]-µ[i];
					sigma[sgi]+=diff*diff*data.getWeight();
				}
			}
			data.next();
		}
		double isNorm=(all1?n-1:n);
		for(int sgi = 0; sgi<sigmaGroups.length; ++sgi) {
			int[] sg = sigmaGroups[sgi];
			for(int i: sg) {
				invSigma[i]=Math.sqrt(sg.length*isNorm/sigma[sgi]);
			}
		}
		for(int i=0; i<invSigma.length; ++i)
			if(invSigma[i]==0)
				System.out.println("No variance group assigned: index "+i);
		return new GaussianWhitener(µ, invSigma);
	}
}
