package org.siquod.neural1.modules;

import java.util.Collections;
import java.util.List;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ActivationSeq;
import org.siquod.neural1.ActivationSet;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;
import org.siquod.neural1.TensorFormat;

/**
 * This layer computes the logarithmized softmax function of its input vector
 * @author bb
 *
 */
public class LogSoftmax implements InOutModule{
	Interface in, out;

	TensorFormat tf;


	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out", in.count);
		if(!in.tf.equals(out.tf))
			throw new IllegalArgumentException("Tensor formats don't match");
		tf=in.tf;
		while(tf.rank>2) {
			tf = tf.flattenIndexAndNext(1);
		}
		if(tf.rank==1) {
			tf = tf.insertUnitIndex(0);
		}
	}

	@Override
	public void allocate(ParamAllocator ia) {
	}

	@Override
	public void share(ParamBlocks ps) {
	}

	@Override
	public ParamBlocks getParamBlocks() {
		return null;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		//out_i = log (exp(in_i) / sum_j exp(in_j))
		//      = in_i - log (sum_j exp(in_j))
		//      = in_i - max - log (sum_j exp(in_j - max))
		if(inst!=null)
			throw new IllegalThreadStateException("A "+getClass().getName()+" module must not be inside a convolution");

		int n0 = tf.dims[0];
		int n1 = tf.dims[1];
		for(ActivationSeq b: as) {
			if(b==null) continue;

			ActivationSet a = b.get(t);
			for(int i0 = 0; i0<n0; ++i0) {
				double sum = 0;
				float max=Float.NEGATIVE_INFINITY;

				for(int i=0; i<n1; ++i){
					float av = a.get(in, tf.index(i0, i));
					if(av>max)
						max=av;
				}
				for(int i=0; i<n1; ++i){
					double av = a.get(in, tf.index(i0, i));			
					sum += Math.exp(av-max);
				}
				float logSum = (float) Math.log(sum);
				for(int i=0; i<n1; ++i){
					//				if(a.get(out, tf.index(i0, i))!=0)
					//					System.out.println();
					//			check2+=Math.exp(a.get(in, i) - logSum);
					float o = a.get(in, tf.index(i0, i)) - max - logSum;
					a.add(out, tf.index(i0, i), o);
					//			check+=Math.exp(a.get(out, i));
				}
			}
			//		System.out.println("check: "+check+" "+check2);
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as,
			ActivationBatch errors, int t, int[] inst) {
		//    d L / d in_i = sum_k (d L / d out_k)*(d out_k / d in_i)
		//    = sum_k (d L / d out_k)*(d (in_k - max - log (sum_j exp(in_j - max)))/(d in_i))
		//    = sum_k (d L / d out_k)*(d (kronnecker_ik - (max==in_i?1:0) - (max==in_i?0:1)*(exp(in_i - max))/(sum_j exp(in_j - max))))
		//    = (d L / d out_i) sum_k (d L / d out_k)*(-(max==in_i?1:0)- (max==in_i?0:1)*exp(in_i - max))/(sum_j exp(in_j - max))))
		int n0 = tf.dims[0];
		int n1 = tf.dims[1];
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;

			ActivationSet a = as.a[b].get(t);
			ActivationSet e = errors.a[b].get(t);
			for(int i0 = 0; i0<n0; ++i0) {
				float odSum=0;
				float sum = 0;
				double max=Double.NEGATIVE_INFINITY;
				int maxi=-1;
				for(int i=0; i<n1; ++i){
					double av = a.get(in, tf.index(i0, i));
					if(av>max){
						max=av;
						maxi=i;
					}
				}
				for(int i=0; i<n1; ++i){
					odSum+=e.get(out, i);
					sum += Math.exp(a.get(in, tf.index(i0, i)) - max);		
				}
				for(int i=0; i<n1; i++){
					if(false && i==maxi){
						e.add(in, i, e.get(out, tf.index(i0, i)) - odSum );
					}else{
						e.add(in, i, e.get(out, tf.index(i0, i)) - odSum * (float)Math.exp(a.get(in, tf.index(i0, i)) - max)/sum );
					}
				}
			}
		}
	}



	////	@Override
	//	public void declareDependencies(Dependencies d) {
	//		d.declare(new InputDependency(in, this, 0));
	//		d.declare(new OutputDependency(this, out));
	//	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}
	//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}
	@Override
	public Interface getIn() {
		return in;
	}
	@Override
	public Interface getOut() {
		return out;
	}
	@Override
	public int dt() {
		return 0;
	}
	@Override
	public int[] shift() {
		return null;
	}
	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
}
