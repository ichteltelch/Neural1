package org.siquod.ml.data;

import java.util.Arrays;


public class PolyInteraction {
	/**
	 * Compute the sum of the n-th simplex numbers from dimension minOrder up to an including dimension maxOrder
	 * @param n
	 * @param minOrder
	 * @param maxOrder
	 * @return
	 */
	public static int simplexNumberSum(int n, int minOrder, int maxOrder) {
		//g n o i = if i>o then 0 else (((n+i)/(i+1))*(1 + g n o (i+1)))
		//ss o n = g n o 1
		long numer = 0;
		long denom = 1;
		for(int i=maxOrder; i>0; --i) {
			if(i>=minOrder)
				numer+=denom;
			numer *= (n+i-1);
			denom *= i;
			long gcd = gcd(numer, denom);
			numer/=gcd;
			denom/=gcd;
		}
		assert denom==1;
		return (int) numer;
		
	}
//	public static int simplexNumberSum(int n, int minOrder, int maxOrder) {
//
//		int ret = 0;
//		for(int i=minOrder; i<=maxOrder; ++i)
//			ret += simplexNumber(n, i);
//		return ret;
//	}
	public static int simplexNumberSum(int n, int maxOrder) {
		//g n o i = if i>=o then 0 else (((n+i)/(i+1))*(1 + g n o (i+1)))
		//ss o n = g n o 0
		long numer = 0;
		long denom = 1;
		for(int i=maxOrder; i>0; --i) {
			numer+=denom;
			numer *= (n+i-1);
			denom *= i;
			long gcd = gcd(numer, denom);
			numer/=gcd;
			denom/=gcd;
		}
		assert denom==1;
		return (int) numer;
		
	}

	public static int simplexNumber(int n, int order) {
		switch(order) {
		case 0: return 1;
		case 1: return n;
		case 2: return (int)(n*(long)(n+1)/2);
		case 3: return (int)(n*(long)(n+1)*(n+2)/6);
		}
		long num = n;
		long den = 1;
		for(int i=1; i<order; ++i) {
			num *= n+i;
			den *= i+1;
			long gcd = gcd(num, den);
			num/=gcd;
			den/=gcd;
		}
		return (int)(num/den);
	}
	/**
	 * Take the {@code n} numbers starting at index {@code inOffset} of array {@code in},
	 * compute all possible products of the involving at least {@code minOrder} 
	 * and at most {@code maxOrder} factors, and write the 
	 * results into the consecutive run of the {@code out} array starting at offset {@code outOffset}
	 * 
	 * @param n How many inputs
	 * @param minOrder Minimum number of factors per product
	 * @param maxOrder Maximum number of factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param out Write results here
	 * @param outOffset offset into outputs
	 * @return The number of results that were written to the outputs array, 
	 * which will be the sum over {@link #simplexNumber(int, int) simplexNumber(n, order)} 
	 * for {@code order} from {@code minOrder} to {@code maxOrder}
	 * 
	 */

	public static int apply(int n, int minOrder, int maxOrder, double[] in, int inOffset, double[] out, int outOffset) {
		int count = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			count+=apply(n, order, in, inOffset, out, outOffset+count, 1);
		}
		return count;
	}

	/**
	 * Take the {@code n} numbers starting at index {@code inOffset} of array {@code in},
	 * compute all possible products of the involving {@code order} factors, and write the 
	 * results into the consecutive run of the {@code out} array starting at offset {@code outOffset}
	 * 
	 * @param n How many inputs
	 * @param order How many factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param out Write results here
	 * @param outOffset offset into outputs
	 * @return The number of results that were written to the outputs array, 
	 * which will be {@link #simplexNumber(int, int) simplexNumber(n, order)}
	 */
	public static int apply(int n, int order, double[] in, int inOffset, double[] out, int outOffset) {
		return apply(n, order, in, inOffset, out, outOffset, 1);
	}

	private static int apply(int n, int order, double[] in, int inOffset, double[] out, int outOffset, double multiplier) {
		if(order<=0) {
			out[outOffset] = multiplier;
			return 1;
		}else if (order==1) {
			for(int i=0; i<n; ++i)
				out[outOffset+i] = in[inOffset+i]*multiplier;
			return n;
		}else {
			int count = 0;
			for(int i=0; i<n; ++i) {
				count+=apply(i+1, order-1, in, inOffset, out, outOffset+count, multiplier*in[inOffset+i]);
			}
			return count;
		}
	}
	/**
	 * Backpropagate gradients through {@link #apply(int, int, int, double[], int, double[], int)}
	 * @param n How many inputs
	 * @param minOrder Minimum number of factors per product
	 * @param maxOrder Maximum number of factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param din Add input gradients here
	 * @param dinOffset offset into dinputs
	 * @param dout Get output gradients from here
	 * @param doutOffset offset into doutputs
	 * @return The number of output gradients that were accessed, 
	 * which will be {@link #simplexNumber(int, int) simplexNumber(n, order)}
	 */
	public static int diffApply(int n, int minOrder, int maxOrder, 
			double[] in, int inOffset, double[] din, int dinOffset,
			double[] dout, int doutOffset) {
		int count = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			count+=diffApply(n, order, in, inOffset, din, dinOffset, dout, doutOffset+count, 1);
		}
		return count;
	}
	
	/**
	 * Backpropagate gradients through {@link #apply(int, int, double[], int, double[], int)}
	 * @param n How many inputs
	 * @param order How many factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param din Add input gradients here
	 * @param dinOffset offset into dinputs
	 * @param dout Get output gradients from here
	 * @param doutOffset offset into doutputs
	 * @return The number of output gradients that were accessed, 
	 * which will be {@link #simplexNumber(int, int) simplexNumber(n, order)}
	 */
	public static int diffApply(int n, int order, double[] in, int inOffset, double[] din, int dinOffset,
			double[] dout, int doutOffset) {
		return diffApply(n, order, in, inOffset, din, dinOffset, dout, doutOffset, 1);
	}
	static int diffApply(int n, int order, double[] in, int inOffset, double[] din, int dinOffset,
			double[] dout, int doutOffset, double multiplier) {
		if(order<=0){
			throw new IllegalArgumentException("order must be positive");
		}else if(order==1) {
			for(int i=0; i<n; ++i)
				din[dinOffset+i] += dout[doutOffset+i]*multiplier;
			return n;
		}else {
			{
				int count = 0;
				for(int i=0; i<n; ++i) {
					count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
							dout, doutOffset+count, multiplier, dinOffset+i);
				}
			}
			{
				int count = 0;
				for(int i=0; i<n; ++i) {
					count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
							dout, doutOffset+count, multiplier*in[inOffset+i]);
				}
				return count;
			}

		}

	}
	static int diffApply(int n, int order, double[] in, int inOffset, double[] din, int dinOffset,
			double[] dout, int doutOffset, double multiplier, int varIndex) {
		if(order<=0){
			throw new IllegalArgumentException("order must be positive");
		}else if(order==1) {
			for(int i=0; i<n; ++i)
				din[varIndex] += dout[doutOffset+i]*in[i]*multiplier;
			return n;
		}else {
			int count = 0;
			for(int i=0; i<n; ++i) {
				count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
						dout, doutOffset+count, multiplier*in[inOffset+i], varIndex);
			}
			return count;

		}

	}

	public static void main(String[] args) {
		int minOrder = 1;
		int maxOrder = 3;
		int n = 27;
		int sum = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			sum += simplexNumber(n, order);
		}
		System.out.println(sum);
		System.out.println(simplexNumberSum(n, minOrder, maxOrder));
	}
	
	/**
	 * Take the {@code n} numbers starting at index {@code inOffset} of array {@code in},
	 * compute all possible products of the involving at least {@code minOrder} 
	 * and at most {@code maxOrder} factors, and write the 
	 * results into the consecutive run of the {@code out} array starting at offset {@code outOffset}
	 * 
	 * @param n How many inputs
	 * @param minOrder Minimum number of factors per product
	 * @param maxOrder Maximum number of factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param out Write results here
	 * @param outOffset offset into outputs
	 * @return The number of results that were written to the outputs array, 
	 * which will be the sum over {@link #simplexNumber(int, int) simplexNumber(n, order)} 
	 * for {@code order} from {@code minOrder} to {@code maxOrder}
	 * 
	 */

	public static int apply(int n, int minOrder, int maxOrder, float[] in, int inOffset, float[] out, int outOffset) {
		int count = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			count+=apply(n, order, in, inOffset, out, outOffset+count, 1);
		}
		return count;
	}

	/**
	 * Take the {@code n} numbers starting at index {@code inOffset} of array {@code in},
	 * compute all possible products of the involving {@code order} factors, and write the 
	 * results into the consecutive run of the {@code out} array starting at offset {@code outOffset}
	 * 
	 * @param n How many inputs
	 * @param order How many factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param out Write results here
	 * @param outOffset offset into outputs
	 * @return The number of results that were written to the outputs array, 
	 * which will be {@link #simplexNumber(int, int) simplexNumber(n, order)}
	 */
	public static int apply(int n, int order, float[] in, int inOffset, float[] out, int outOffset) {
		return apply(n, order, in, inOffset, out, outOffset, 1);
	}
	

	private static int apply(int n, int order, float[] in, int inOffset, float[] out, int outOffset, float multiplier) {
		if(order<=0) {
			out[outOffset] = multiplier;
			return 1;
		}else if (order==1) {
			for(int i=0; i<n; ++i)
				out[outOffset+i] = in[inOffset+i]*multiplier;
			return n;
		}else {
			int count = 0;
			for(int i=0; i<n; ++i) {
				count+=apply(i+1, order-1, in, inOffset, out, outOffset+count, multiplier*in[inOffset+i]);
			}
			return count;
		}
	}
	/**
	 * Backpropagate gradients through {@link #apply(int, int, int, float[], int, float[], int)}
	 * @param n How many inputs
	 * @param minOrder Minimum number of factors per product
	 * @param maxOrder Maximum number of factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param din Add input gradients here
	 * @param dinOffset offset into dinputs
	 * @param dout Get output gradients from here
	 * @param doutOffset offset into doutputs
	 * @return The number of output gradients that were accessed, 
	 * which will be {@link #simplexNumber(int, int) simplexNumber(n, order)}
	 */
	public static int diffApply(int n, int minOrder, int maxOrder, 
			float[] in, int inOffset, float[] din, int dinOffset,
			float[] dout, int doutOffset) {
		int count = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			count+=diffApply(n, order, in, inOffset, din, dinOffset, dout, doutOffset+count, 1);
		}
		return count;
	}
	
	/**
	 * Backpropagate gradients through {@link #apply(int, int, float[], int, float[], int)}
	 * @param n How many inputs
	 * @param order How many factors per product
	 * @param in Get inputs from here
	 * @param inOffset offset into inputs
	 * @param din Add input gradients here
	 * @param dinOffset offset into dinputs
	 * @param dout Get output gradients from here
	 * @param doutOffset offset into doutputs
	 * @return The number of output gradients that were accessed, 
	 * which will be {@link #simplexNumber(int, int) simplexNumber(n, order)}
	 */
	public static int diffApply(int n, int order, float[] in, int inOffset, float[] din, int dinOffset,
			float[] dout, int doutOffset) {
		return diffApply(n, order, in, inOffset, din, dinOffset, dout, doutOffset, 1);
	}
	static int diffApply(int n, int order, float[] in, int inOffset, float[] din, int dinOffset,
			float[] dout, int doutOffset, float multiplier) {
		if(order<=0){
			throw new IllegalArgumentException("order must be positive");
		}else if(order==1) {
			for(int i=0; i<n; ++i)
				din[dinOffset+i] += dout[doutOffset+i]*multiplier;
			return n;
		}else {
			{
				int count = 0;
				for(int i=0; i<n; ++i) {
					count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
							dout, doutOffset+count, multiplier, dinOffset+i);
				}
			}
			{
				int count = 0;
				for(int i=0; i<n; ++i) {
					count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
							dout, doutOffset+count, multiplier*in[inOffset+i]);
				}
				return count;
			}

		}

	}
	static int diffApply(int n, int order, float[] in, int inOffset, float[] din, int dinOffset,
			float[] dout, int doutOffset, float multiplier, int varIndex) {
		if(order<=0){
			throw new IllegalArgumentException("order must be positive");
		}else if(order==1) {
			for(int i=0; i<n; ++i)
				din[varIndex] += dout[doutOffset+i]*in[i]*multiplier;
			return n;
		}else {
			int count = 0;
			for(int i=0; i<n; ++i) {
				count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
						dout, doutOffset+count, multiplier*in[inOffset+i], varIndex);
			}
			return count;

		}

	}

	public static void main2(String[] args) {
		float[] primes= {
				2,3,5,7
		};
		float[] out = new float[100];
		apply(4, 3, primes, 0, out, 0);
		System.out.println(Arrays.toString(out));

	}
	public static long gcd(long a, long b) {
		if(a<0)a=-a;
		if(b<0)b=-b;
		if(a>b) {
			long t=a;
			a=b;
			b=t;
		}
		while(a!=0) {
			long t = b%a;
			b=a;
			a=t;
		}
		return b;
	}
}
