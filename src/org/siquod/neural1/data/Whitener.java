package org.siquod.neural1.data;

import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.Consumer;



public interface Whitener {
	void whiten(double[] in, double[] out);
	void whiten(float[] in, float[] out);
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
		public void whiten(float[] in, float[] out) {
			for(int i=0; i<µ.length; ++i) {
				out[i] = (float)((in[i]-µ[i])*invSigma[i]);
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
	static class MultivariateGaussianWhitener implements Whitener{
		public final double[] µ;
		public final double[][] invSigma;
		public MultivariateGaussianWhitener(double[] µ, double[][] invSigma) {
			this.µ=µ;
			this.invSigma=invSigma;
		}
		@Override
		public void whiten(double[] in, double[] out) {
			for(int i=0; i<µ.length; ++i) {
				double sum = 0;
				for(int j=0; j<µ.length; ++j){
					sum += (in[j]-µ[j])*invSigma[i][j];
				}
				out[i] = sum;
			}
		}
		@Override
		public void whiten(float[] in, float[] out) {
			for(int i=0; i<µ.length; ++i) {
				float sum = 0;
				for(int j=0; j<µ.length; ++j){
					sum += (in[j]-µ[j])*invSigma[i][j];
				}
				out[i] = sum;
			}
		}
		@Override
		public int dim() {
			return µ.length;
		}

	}
	public static GaussianWhitener gaussianForInputs(TrainingBatchCursor data) {
		int dim = data.inputCount();
		Consumer<double[]> give = data::giveInputs;
		return makeGaussianWhitener(data, dim, give);
	}
	public static MultivariateGaussianWhitener multivariateGaussianForInputs(TrainingBatchCursor data, double regularization, boolean preWhiten) {
		int dim = data.inputCount();
		Consumer<double[]> give = data::giveInputs;
		return makeMultivatiateGaussianWhitener(data, dim, give, regularization, preWhiten);
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
	static MultivariateGaussianWhitener makeMultivatiateGaussianWhitener(TrainingBatchCursor data, int dim, Consumer<double[]> give, double regularization, boolean preWhiten) {
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


		double[] sigma0 = new double[dim];
		double[] invSigma0 = sigma0;
		double isNorm=(all1?n-1:n);

		if(preWhiten) {
			data.reset();
			while(!data.isFinished()) {
				give.accept(buffer);
				double w = data.getWeight();
				for(int i=0; i<dim; ++i) {
					double diff=buffer[i]-µ[i];
					sigma0[i]+=diff*diff*w;
				}
				data.next();
			}
			for(int i=0; i<dim; ++i)
				invSigma0[i]=Math.sqrt(isNorm/(sigma0[i]+1e-8));
		}else {
			Arrays.fill(invSigma0, 1);

		}


		double[][] sigma = new double[dim][dim];
		data.reset();
		while(!data.isFinished()) {
			give.accept(buffer);
			for(int i=0; i<dim; ++i) 
				buffer[i] = (buffer[i]-µ[i])*invSigma0[i];
			double weight = data.getWeight();
			for(int i=0; i<dim; ++i) {
				double biw = buffer[i]*weight;
				for(int j=i; j<dim; ++j) 
					sigma[i][j]+=biw*buffer[j];
			}

			data.next();
		}



		for(int i=0; i<dim; ++i) {

			sigma[i][i] = sigma[i][i]/isNorm;
			for(int j=i+1; j<dim; ++j) {
				sigma[j][i] = sigma[i][j] /= isNorm;
			}
		}
		//		double[][] sigmaCopy = copy(dim, sigma, null);
		double[][] invSigma = new double[dim][dim];
		double[][] basis = new double[dim][dim];
		double[] eigenvalues = new double[dim];
		eigen(dim, sigma, sigma, null, basis, eigenvalues, 100);
		//		System.out.println();
		for(int i=0; i<dim; ++i) {
			eigenvalues[i] = 1/Math.max(Math.sqrt(Math.max(0, eigenvalues[i])), regularization);
		}
		deDiagonalize(dim, invSigma, basis, eigenvalues, new double[dim][dim]);
		transposeInPlace(dim, basis);
		double[] sigma2 = new double[dim+0];
		data.reset();
		while(!data.isFinished()) {
			give.accept(buffer);
			double w = data.getWeight();
			for(int i=0; i<dim; ++i) 
				buffer[i] = (buffer[i]-µ[i])*invSigma0[i];
			for(int i=0; i<µ.length; ++i) {
				double sum = 0;
				for(int j=0; j<µ.length; ++j){
					sum += buffer[j]*invSigma[i][j];
				}
				sigma2[i] += w * sum*sum;
			}
			data.next();
		}
		double[] invSigma2 = sigma2;
		for(int i=0; i<dim; ++i)
			invSigma2[i]=Math.sqrt(isNorm/(sigma2[i]+1e-8));
		for(int i=0; i<dim; ++i) {
			for(int j=0; j<dim; ++j) {
				invSigma[i][j] *= invSigma2[i]*invSigma0[j];
			}

		}


		//		if(true) {
		//			double[][] check = new double[dim][dim];
		//			mul(sigmaCopy, invSigma, check);
		//			System.out.println();
		//		}
		return new MultivariateGaussianWhitener(µ, invSigma);
	}
	static double[][] copy(int dim, double[][] mat, double[][] matCopy) {
		if(matCopy==null)
			matCopy = new double[dim][dim];
		for(int i=0; i<dim; ++i) {
			System.arraycopy(mat[i], 0, matCopy[i], 0, dim);
		}
		return matCopy;
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

	public static void invert(int dim, double[][] rows, double[][] inv) {
		setUnit(dim, inv);
		for(int r=0; r<dim; r++){
			{
				int mpr=r;
				for(int pr=r+1; pr<dim; pr++)
					if(Math.abs(rows[pr][r])>Math.abs(rows[mpr][r]))
						mpr=pr;
				if(mpr!=r){
					double[] tr=rows[r];
					rows[r]=rows[mpr];
					rows[mpr]=tr;

					tr=inv[r];
					inv[r]=inv[mpr];
					inv[mpr]=tr;
				}
			}
			{
				double[] row=rows[r];
				double pivot_ = row[r];
				for(int r2=r+1; r2<dim; r2++){
					double[] row2=rows[r2];
					double head = row2[r];

					double size = Math.max(Math.abs(head),Math.abs(pivot_));
					double digits = Math.round(Math.log(size)/Math.log(2));
					double renorm = Math.pow(2, -Math.round(digits));

					double pivot=pivot_*renorm;
					head*=renorm;
					row2[r] = 0;
					for(int col=r+1; col<dim; col++){
						row2[col] = row2[col] * pivot - row[col] * head;
					}
					for(int n=0; n<dim; n++){
						inv[r2][n] = inv[r2][n] * pivot - inv[r][n] * head;
					}
				}
			}
		}
		for(int r=dim-1; r>=0; r--){
			double norm=1/rows[r][r];
			for(int c=r; c<dim; c++)
				rows[r][c]*=norm;
			for(int c=0; c<dim; c++)
				inv[r][c] *= norm;
			for(int r2=r-1; r2>=0; r2--){
				double factor = -rows[r2][r];
				rows[r2][r]=0;
				for(int c=0; c<dim; c++)
					inv[r2][c] += inv[r][c]*factor;
			}
		}


	}
	public static void setUnit(int dim, double[][] inv) {
		for(int i=0; i<dim; ++i) {
			double[] row = inv[i];
			Arrays.fill(row, 0);
			row[i]=1;
		}
	}
	public static void mul(double[][] m1, double[][] m2, double[][] out) {
		for(int row=0; row<m1.length; row++){
			for(int col=0; col<m2[0].length; col++){
				double s=0;
				for(int i=0; i<m2.length; i++)
					s += m1[row][i]*m2[i][col];
				out[row][col]=s;
			}
		}
	}

	//	public static void main(String[] args) {
	//		double[][] is = {
	//				{1,0},
	////				{-1,0},
	//				{0,2},
	////				{0,-2},
	//				{1,1},
	////				{-1,-1},
	//		};
	//		double[] ws = new double[is.length];
	//		Arrays.fill(ws, 2);
	//		TrainingBatchCursor test = new TrainingBatchCursor.RamBuffer(is, new double[is.length][0], ws, 0, 2, 0);
	//		double regularization=0;
	//		Whitener w1 = multivariateGaussianForInputs(test, regularization);
	//		TrainingBatchCursor whitened = test.whitened(w1, null);
	//		Whitener w2 = multivariateGaussianForInputs(whitened, regularization);
	//
	//	}

	public static void eigen(int dim, double[][] of, double[][] use, double[][] basisGuess, double[][] outBasis, double[] values, int iter){
		if(use == null)
			use=copy(dim, of, null);
		else if(use!=of)
			use=copy(dim, of, use);

		if(basisGuess==null)
			setUnit(dim, outBasis);
		else{
			if(basisGuess==outBasis){
				preDiagonalize(dim, use, basisGuess, new double[dim][dim]);
			}else{
				preDiagonalize(dim, use, basisGuess, outBasis);
				copy(dim, basisGuess, outBasis);
			}
		}
		dynDiagonalize(dim, use, outBasis, iter);
		for(int i=0; i<dim; i++)
			values[i]=use[i][i];

	}
	public static double[][] dynDiagonalize(int dim, double[][] e, double[][] outBasis, int iter){
		double[][] rot=outBasis;
		for(int itt=0; itt<iter; itt++){
			for(int _i=0; _i<dim; _i++)
				for(int _j=_i+1; _j<dim; _j++){
					int i, j;
					if((iter&1)==0){
						i=_i;
						j=_j;						
					}else{
						i=dim-_i-1;
						j=dim-_j-1;						
					}
					//set e[i][j] to 0 while maintaining that M = rot × this × rot^-1
					double sin, cos;

					double v=e[j][j], y=e[i][i], w=e[j][i];
					//					double wq=w*w, vy=v*y, vpy=v+y;
					@SuppressWarnings("unused")
					double l1, l2;
					double sqw=w*w;
					double nph=(v+y)*.5, q = v*y-sqw, detv = (nph*nph-q);
					if(detv<=0)
						continue;
					double sqrtdeth=sqrt(detv);
					l1 = nph + sqrtdeth;
					l2 = nph - sqrtdeth;


					double yl=y-l1;
					double vl=v-l1;
					double norm1=sqw+yl*yl;
					double norm2=sqw+vl*vl;
					if(norm1>norm2){
						norm1=(double) (1/sqrt(norm1));
						sin = w*norm1;
						cos = yl*norm1;
					}else{
						norm2=(double) (1/sqrt(norm2));
						cos = w*norm2;
						sin = vl*norm2;
					}



					//					double t1= (v-y)*cos*sin + w*cos*cos - w*sin*sin ;
					//					double t2=  cos*cos +   sin*sin ;

					rotR(dim, rot, j, i, -cos, -sin);
					rotL(dim, e, j, i, -cos, sin);
					rotR(dim, e, j, i, -cos, -sin);
					e[j][i]=e[i][j]=0;
				}
		}
		return e;
	}
	public static double[][] preDiagonalize(int dim, double[][] on, double[][] basis, double[][] t) {
		//basis^T x this x basis = (this^T x basis)^T x basis
		//transpose(); //unnecessary: is symmetric
		mul(on, basis, t);
		transposeInPlace(dim, t);
		mul(t, basis, on);
		return on;
	}
	public static void transposeInPlace(int dim, double[][] e){
		for(int i=0; i<dim; i++)
			for(int j=i+1; j<dim; j++){
				double t=e[i][j];
				e[i][j]=e[j][i];
				e[j][i]=t;
			}
	}
	public static void rotR(int dim, double[][] e, int i, int j, double cos, double sin){
		for(int a=0; a<dim; a++){
			double aj=e[a][j];
			double ai=e[a][i];
			e[a][i] = ai*cos - sin*aj;  
			e[a][j] = aj*cos + sin*ai;
		}
	}


	public static void rotL(int dim, double[][] e, int i, int j, double cos, double sin){
		for(int a=0; a<dim; a++){
			double ja=e[j][a];
			double ia=e[i][a];
			e[i][a] = cos*ia + sin*ja;  
			e[j][a] = cos*ja - sin*ia;
		}

	}
	public static double[][] deDiagonalize(int dim, double[][] e, double[][] basis, double[] diag, double[][] t) {
		if(t==null)
			t= new double[dim][dim];
		for(int i=0; i<dim; ++i) {
			double ev = diag[i];
			for(int j=0; j<dim; ++j) {
				t[j][i] = basis[j][i] * ev;
			} 
		}		
		mulTranspose(dim, t, basis, e);
		return e;
	}
	public static double[][] mulTranspose(double dim, double[][] m1, double[][] m2, double[][] out){
		for(int i=0; i<dim; i++)
			for(int j=0; j<dim; j++){
				double sum=0;
				for(int k=0; k<dim; k++)
					sum+=m1[i][k]*m2[j][k];
				out[i][j]=sum;
			}
		return out;
	}
	@SuppressWarnings("unused")
	public static void mainz(String[] args) {
		int m = 10;
		int n = 1000;
		double[][] vs = new double[n][];
		double[][] pvs = new double[m][];
		Random r = new Random(123);
		for(int i=0; i<n; ++i) {
			for(int j=0; j<m; ++j) {
				double x = r.nextGaussian();
				double y = r.nextGaussian();
				double z = r.nextGaussian();
				double norm = Math.sqrt(x*x+y*y+z*z);
				x/=norm;
				y/=norm;
				z/=norm;
				//				if(y<-0.5) {
				//					--j;
				//					continue;
				//				}
				//				if(Math.abs(x-0.1)>0.6) {
				//					--j;
				//					continue;
				//				}
				double[] feat0 = {
						x, y, z,   					//0 1 2 
						x*x, x*y, x*z,              //3 4 5
						y*y, y*z,                   //6 7
						//z*z,                        //8
						x*x*x, x*x*y, /*x*x*z,*/        //9 10 11
						/*x*y*y,*/ x*y*z,               //12 13
						x*z*z,                      //14
						y*y*y, y*y*z,               //15 16
						/*y*z*z,*/                      //17
						z*z*z,                      //18
				};
				double[] feat1 = {
						x, x*x, x*y, y*y, x*x*x, x*x*y, x*z*z, y*y*y
				};
				double[] feat2 = {
						x, y, z,   					//0 1 2 
						x*x, x*y, x*z,              //3 4 5
						y*y, y*z,                   //6 7
						//z*z,                        //8
						/*x*x*x,*/ x*x*y, x*x*z,        //9 10 11
						x*y*y, x*y*z,               //12 13
						x*z*z,                      //14
						/*y*y*y, */y*y*z,               //15 16
						y*z*z,                      //17
						//z*z*z,                      //18
				};
				pvs[j]=feat0;
			}
			int l = pvs[0].length;
			double[] a = new double[l];
			for(int j=0; j<m; ++j) {
				for(int k=0; k<l; ++k) {
					a[k] += pvs[j][k];
				}
			}
			for(int k=0; k<l; ++k) {
				a[k] /= m;
			}
			vs[i]=a;
		}

		TrainingBatchCursor c = new TrainingBatchCursor.RamBuffer(vs, new double[n][0], null, 0, vs[0].length, 0);
		Whitener w = Whitener.multivariateGaussianForInputs(c, 0.0001, true);
	}
}
