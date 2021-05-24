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

/**
 * This layer computes the logarithmized softmax function of its input vector
 * @author bb
 *
 */
public class LogSoftmax implements InOutModule{
	Interface in, out;
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out", in.count);
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

		for(ActivationSeq b: as.a) {
			if(b==null) continue;

			ActivationSet a = b.get(t);
			double sum = 0;
			float max=Float.NEGATIVE_INFINITY;
			for(int i=0; i<in.count; ++i){
				float av = a.get(in, i);
				if(av>max)
					max=av;
			}
			for(int i=0; i<in.count; ++i){
				double av = a.get(in, i);			
				sum += Math.exp(av-max);
			}
			float logSum = (float) Math.log(sum);
			for(int i=0; i<in.count; ++i){
				if(a.get(out, i)!=0)
					System.out.println();
				//			check2+=Math.exp(a.get(in, i) - logSum);
				float o = a.get(in, i) - max - logSum;
				a.add(out, i, o);
				//			check+=Math.exp(a.get(out, i));
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
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;

			ActivationSet a = as.a[b].get(t);
			ActivationSet e = errors.a[b].get(t);
			float odSum=0;
			float sum = 0;
			double max=Double.NEGATIVE_INFINITY;
			int maxi=-1;
			for(int i=0; i<in.count; ++i){
				double av = a.get(in, i);
				if(av>max){
					max=av;
					maxi=i;
				}
			}
			for(int i=0; i<in.count; ++i){
				odSum+=e.get(out, i);
				sum += Math.exp(a.get(in, i) - max);		
			}
			for(int i=0; i<in.count; i++){
				if(false && i==maxi){
					e.add(in, i, e.get(out, i) - odSum );
				}else{
					e.add(in, i, e.get(out, i) - odSum * (float)Math.exp(a.get(in, i) - max)/sum );
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
