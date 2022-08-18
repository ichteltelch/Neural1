package org.siquod.neural1.data;

import java.util.Arrays;

/**
 * This class is a copy of {@link PolyInteraction} where doubles are replaced by Strings
 * so as to verify that everything is correct.
 * @author bb
 *
 */
public class PolyInteractionString {

	public static int simplexNumber(int n, int order) {
		long num = n;
		long den = 1;
		for(int i=1; i<order; ++i) {
			num *= n+i;
			den *= i+1;
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

	public static int apply(int n, int minOrder, int maxOrder, String[] in, int inOffset, String[] out, int outOffset) {
		int count = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			count+=apply(n, order, in, inOffset, out, outOffset+count, "");
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
	public static int apply(int n, int order, String[] in, int inOffset, String[] out, int outOffset) {
		return apply(n, order, in, inOffset, out, outOffset, "");
	}

	private static int apply(int n, int order, String[] in, int inOffset, String[] out, int outOffset, String multiplier) {
		if(order<=0) {
			out[outOffset] = multiplier;
			return 1;
		}else if (order==1) {
			for(int i=0; i<n; ++i)
				out[outOffset+i] = multiplier+" "+in[inOffset+i];
			return n;
		}else {
			int count = 0;
			for(int i=0; i<n; ++i) {
				count+=apply(i+1, order-1, in, inOffset, out, outOffset+count, multiplier+" "+in[inOffset+i]);
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
			String[] in, int inOffset, String[] din, int dinOffset,
			String[] dout, int doutOffset) {
		int count = 0;
		for(int order = minOrder; order<=maxOrder; ++order) {
			count+=diffApply(n, order, in, inOffset, din, dinOffset, dout, doutOffset+count, "");
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
	public static int diffApply(int n, int order, String[] in, int inOffset, String[] din, int dinOffset,
			String[] dout, int doutOffset) {
		return diffApply(n, order, in, inOffset, din, dinOffset, dout, doutOffset, "");
	}
	static int diffApply(int n, int order, String[] in, int inOffset, String[] din, int dinOffset,
			String[] dout, int doutOffset, String multiplier) {
		if(order<=0){
			throw new IllegalArgumentException("order must be positive");
		}else if(order==1) {
			for(int i=0; i<n; ++i)
				din[dinOffset+i] += " + " + multiplier + " " + dout[doutOffset+i];
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
							dout, doutOffset+count, multiplier+" "+in[inOffset+i]);
				}
				return count;
			}

		}

	}
	static int diffApply(int n, int order, String[] in, int inOffset, String[] din, int dinOffset,
			String[] dout, int doutOffset, String multiplier, int varIndex) {
		if(order<=0){
			throw new IllegalArgumentException("order must be positive");
		}else if(order==1) {
			for(int i=0; i<n; ++i)
				din[varIndex] += " + "+multiplier+" "+in[i]+" "+dout[doutOffset+i];
			return n;
		}else {
			int count = 0;
			for(int i=0; i<n; ++i) {
				count+=diffApply(i+1, order-1, in, inOffset, din, dinOffset, 
						dout, doutOffset+count, multiplier+" "+in[inOffset+i], varIndex);
			}
			return count;

		}

	}

	public static void main(String[] args) {
		String[] vars= {
				"a", "b", "c", "d"
		};
		String[] dsums = {
				"0", "0", "0", "0"
		};
		String[] out = new String[100];
		int n = apply(4, 2, 3, vars, 0, out, 0);
		for(int i=0; i<n; ++i) {
			out[i]="(ð"+out[i]+")";
		}
		System.out.println(Arrays.toString(out));
		diffApply(4, 2, 3, vars, 0, dsums, 0, out, 0);
		System.out.println(dsums[0]);
		System.out.println(dsums[1]);
		System.out.println(dsums[2]);
		System.out.println(dsums[3]);

	}
}
