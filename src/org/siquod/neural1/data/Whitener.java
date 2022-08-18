package org.siquod.neural1.data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
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
		/**
		 * Given logistic regression parameters trained on whitened data, transform them into equivalent
		 * parameters for the colorized version of the data
		 * @param pIn
		 * @param pOut
		 */
		public double[] colorizeLogisticRegressionParams(double[] pIn, double[] pOut) {
			/**
			 * Need to solve:
			 * pIn[0..last-1]·whitened + pIn[last] = pOut[0..last-1]·colored + pOut[last]
			 * where whitened = (diag(invSigma))(colored - µ)
			 * so: 
			 * pIn[0..last-1]·(diag(invSigma))(colored - µ) + pIn[last] = pOut[0..last-1]·colored + pOut[last]
			 * pIn[0..last-1]·(diag(invSigma)) colored - pIn[0..last-1]·(diag(invSigma)) µ + pIn[last] = pOut[0..last-1]·colored + pOut[last]
			 * pOut[0..last-1] = (diag(invSigma))·pIn[0..last-1] 
			 * pOut[last] = pIn[last] - pIn[0..last-1]·(diag(invSigma)) µ
			 */
			if(pOut==null)
				pOut=new double[pIn.length];
			int last = pIn.length-1;
			double acc = pIn[last];
			for(int i=0; i<last; ++i) {
				double s = pIn[i]*invSigma[i];
				pOut[i]=s;
				acc -= s*µ[i];
			}
			pOut[last]=acc;
			return pOut;
			
		}
		/**
		 * Given logistic regression parameters that work on colored data, transform them into equivalent
		 * parameters for the whitened version of the data
		 * @param pIn
		 * @param pOut
		 */
		public double[] whitenLogisticRegressionParams(double[] pIn, double[] pOut) {
			/**
			 * Need to solve:
			 * pOut[0..last-1]·whitened + pOut[last] = pIn[0..last-1]·colored + pIn[last]
			 * where whitened = (diag(invSigma))(colored - µ)
			 * so: 
			 * colored = diag(sigma) whitened + µ 
			 * pOut[0..last-1]·whitened + pOut[last] = pIn[0..last-1]·(diag(sigma) whitened + µ) + pIn[last]
			 * pOut[0..last-1]·whitened + pOut[last] = pIn[0..last-1]·diag(sigma) whitened + pIn[0..last-1]·µ + pIn[last]
			 * pOut[0..last-1] = (diag(sigma))·pIn[0..last-1] 
			 * pOut[last] = pIn[last] + pIn[0..last-1]·µ
			 */
			if(pOut==null)
				pOut=new double[pIn.length];
			int last = pIn.length-1;
			double acc = pIn[last];
			for(int i=0; i<last; ++i) {
				double pInI = pIn[i];
				pOut[i]=pInI/invSigma[i];
				acc += pInI*µ[i];
			}
			pOut[last]=acc;
			return pOut;
			
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
		Consumer<double[]> give = data::giveInputs;
		return makeGaussianWhitener(data, dim, give);
	}
	public static GaussianWhitener gaussianForOutputs(TrainingBatchCursor data) {
		int dim = data.outputCount();
		Consumer<double[]> give = data::giveOutputs;
		return makeGaussianWhitener(data, dim, give);
	}
	public static GaussianWhitener gaussianForInputs(TrainingBatchCursor[] data, ExecutorService exec) throws InterruptedException {
		int dim = data[0].inputCount();
		@SuppressWarnings("unchecked")
		Consumer<double[]>[] give = new Consumer[data.length];
		for(int i=0; i<data.length; ++i) {
			TrainingBatchCursor d = data[i];
			if(d.inputCount()!=dim)
				throw new IllegalArgumentException("Dimension mismatch");
			give[i] = d::giveInputs;

		}
		return makeGaussianWhitener(data, dim, give, exec);
	}
	public static GaussianWhitener gaussianForOutputs(TrainingBatchCursor[] data, ExecutorService exec) throws InterruptedException {
		int dim = data[0].outputCount();
		@SuppressWarnings("unchecked")
		Consumer<double[]>[] give = new Consumer[data.length];
		for(int i=0; i<data.length; ++i) {
			TrainingBatchCursor d = data[i];
			if(d.outputCount()!=dim)
				throw new IllegalArgumentException("Dimension mismatch");
			give[i] = d::giveOutputs;
		}
		return makeGaussianWhitener(data, dim, give, exec);	
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
			sigma[i]=Math.sqrt(isNorm/(sigma[i]+1e-8));
		return new GaussianWhitener(µ, sigma);
	}
	static class MutBool{boolean val; MutBool(boolean v){val=v;}}
	static class MutDouble{double val; MutDouble(double v){val=v;}}
	static GaussianWhitener makeGaussianWhitener(TrainingBatchCursor[] datas, int dim, Consumer<double[]>[] gives, ExecutorService exec) throws InterruptedException {
		int threads = datas.length;
		

		double[][] µs=new double[threads][dim];
		double[][] buffers=new double[threads][dim];
		MutBool all1s=new MutBool(true);
		MutDouble ns = new MutDouble(0);
		
		Runnable[] jobs = new Runnable[threads];
		for(int jobId = 0; jobId<threads; ++jobId) {
			TrainingBatchCursor data = datas[jobId];
			double[] µ = µs[jobId];
			double[] buffer=buffers[jobId];
			Consumer<double[]> give = gives[jobId];
			jobs[jobId] = ()->{
				data.reset();
				boolean all1=true;
				double n=0;
	
				while(!data.isFinished()) {
					double w = data.getWeight();
					all1 &= w==1;
					n+=w;
					give.accept(buffer);
					for(int i=0; i<dim; ++i)
						µ[i]+=w*buffer[i];
					data.next();
				}	
				synchronized (ns) {
					all1s.val&=all1;
					ns.val+=n;
				}
			};
		}
		parallel(exec, jobs);
		boolean all1 = all1s.val;
		double n=ns.val;

		double[] µ=new double[dim];
		for(double[] b: µs) {
			for(int i=0; i<µ.length; ++i)
				µ[i] += b[i];
		}
		
		
		double µNorm=1/n;
		for(int i=0; i<dim; ++i)
			µ[i]*=µNorm;
		double[][] sigmas = new double[threads][dim];

		for(int jobId = 0; jobId<threads; ++jobId) {
			double[] sigma = sigmas[jobId];
			TrainingBatchCursor data = datas[jobId];
			double[] buffer=buffers[jobId];
			Consumer<double[]> give = gives[jobId];
			jobs[jobId] = ()->{
				data.reset();
				while(!data.isFinished()) {
					give.accept(buffer);
					for(int i=0; i<dim; ++i) {
						double diff=buffer[i]-µ[i];
						sigma[i]+=diff*diff*data.getWeight();
					}
					data.next();
				}
			};
		}
		parallel(exec, jobs);
		double[] sigma = new double[dim];
		for(double[] b: sigmas) 
			for(int i=0; i<sigma.length; ++i)
				sigma[i] += b[i];

		double isNorm=(all1?n-1:n);
		for(int i=0; i<dim; ++i)
			sigma[i]=Math.sqrt(isNorm/(sigma[i]+1e-8));
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
				invSigma[i]=sigma[sgi]==0?1:Math.sqrt(sg.length*isNorm/sigma[sgi]);
			}
		}
		for(int i=0; i<invSigma.length; ++i)
			if(invSigma[i]==0)
				System.out.println("No variance group assigned: index "+i);
		return new GaussianWhitener(µ, invSigma);
	}
//	public static void main(String[] args) throws InterruptedException {
//		int n = 1000000;
//		double[][] inputs  = new double[n][2];
//		double[] outputs = new double[n];
//		double[] idealParams = {5, 2, -0.5};
//		double sigma0 = 2;
//		double sigma1 = 3;
//		double µ0 = -.1;
//		double µ1 = -.2;
//		Random r = new Random(42);
//		for(int i=0; i<n; ++i) {
//			double[] inp = inputs[i];
//			inp[0] = µ0 + sigma0*r.nextGaussian();
//			inp[1] = µ1 + sigma1*r.nextGaussian();
//			double p = LogisticRegression.classify(inp, idealParams);
//			outputs[i] = p>r.nextDouble()?1:0;
//		}
//		RandomAccess data = TrainingBatchCursor.forArray(inputs, outputs);
//		GaussianWhitener whitener = Whitener.gaussianForInputs(data.split(4));
//		RandomAccess whiteData = data.whitened(whitener, null);
//		double[] whiteParams = new double[3];
//		LogisticRegression.newton(whiteData, whiteParams, 1e-10, 1000, 0);
//		double[] colorParams = new double[3];
//		whitener.colorizeLogisticRegressionParams(whiteParams, colorParams);
//		double[] whiteParams2 = new double[3];
//		whitener.whitenLogisticRegressionParams(colorParams, whiteParams2);
//		System.out.println(Arrays.toString(whiteParams));
//		System.out.println(Arrays.toString(whiteParams2));
//
//	}
	public static void parallel(ExecutorService exec, Runnable... rs) throws InterruptedException {
		ArrayList<Future<?>> fs=new ArrayList<>(rs.length);
		for(Runnable run: rs) {
			if(run!=null)
				fs.add(exec.submit(run));
		}
		joinAll(fs);
	}
	public static void joinAll(Collection<? extends Future<?>> chunks) throws InterruptedException {
		try {
			for(Future<?> f: chunks)
				f.get();
		} catch (ExecutionException e) {
			throwCause(e);
		}finally {
			for(Future<?> f: chunks)
				if(f!=null)
					f.cancel(true);	
		}
	}
	/**
	 * This throws the exceptions cause if it is either a {@link RuntimeException} or an {@link Error}
	 * @param <T>
	 * @param e
	 * @return
	 * @throws IllegalArgumentException if the cause is <code>null</code> or a checked {@link Exception}
	 */
	public static <T> T throwCause(Exception e) {
		Throwable cause = e.getCause();
		if(cause instanceof RuntimeException)
			throw (RuntimeException)cause;
		if(cause instanceof Error)
			throw (Error)cause;
		else if(cause==null) 
			throw new IllegalArgumentException("throwCause(): Argument has no cause", e);
		else 
			throw new IllegalArgumentException("throwCause(): Cause of the argument is checked", e);
	}
}
