package org.siquod.ml.neural1.modules;

import java.util.Collections;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ActivationSeq;
import org.siquod.ml.neural1.ActivationSet;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;
import org.siquod.ml.neural1.modules.loss.LossGroup;

/**
 * This layer computes the logarithmized softmax function of its input vector
 * 
 * You can divide the vector into several {@link LossGroup}s which will be treated separately
 * @author bb
 *
 */
public class LogSoftmax implements InOutModule{
	Interface in, out;

	TensorFormat tf;

	LossGroup[] lgs;

	public LogSoftmax(LogSoftmax copyThis) {
		this.in=copyThis.in;
		this.out=copyThis.out;
		this.tf=copyThis.tf;
		this.lgs=copyThis.lgs;
	}
	@Override
	public LogSoftmax copy() {
		return this;
	}
	public LogSoftmax(LossGroup... lgs) {
		this.lgs=lgs;
	}

	public LogSoftmax() {

	}

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
		int channels = tf.channels();
		if(lgs==null)
			lgs=LossGroup.makeDefault(channels);
		else {
			int lgl = lgs[lgs.length-1].end;
			if(lgl > channels) {
				throw new IllegalStateException("There are not enough channels for the specified LossGroups");
			}else if(lgl < channels) {
				System.err.println("Warning: There are more channels than specified by the LossGroups");
			}
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
		for(ActivationSeq b: as) {
			if(b==null) continue;

			ActivationSet a = b.get(t);
			for(int bri = 0; bri<n0; ++bri) {
				for(LossGroup lg: lgs) {
					int iStart = lg.start;
					int iEnd = lg.end;
					boolean singleton = lg.isSingleton();
					double sum;
					float max;

					if(singleton) {
						max = 0;
						sum = 1;
						int i = lg.start;
						int index = tf.index(bri, i);
						float raw = a.get(in, index);
						float o = (float) (1/(1+Math.exp(-raw)));
						if(!Float.isFinite(o))
							System.out.println();
						a.add(out, index, o);
						//			check+=Math.exp(a.get(out, i));

					}else {
						max=Float.NEGATIVE_INFINITY;
						for(int i=iStart; i<iEnd; ++i){
							float av = a.get(in, tf.index(bri, i));
							if(av>max)
								max=av;
						}
						sum = 0;
						for(int i=iStart; i<iEnd; ++i){
							double av = a.get(in, tf.index(bri, i));			
							sum += Math.exp(av-max);
						}
						float logSum = (float) Math.log(sum);
						for(int i=iStart; i<iEnd; ++i){
							//				if(a.get(out, tf.index(i0, i))!=0)
							//					System.out.println();
							//			check2+=Math.exp(a.get(in, i) - logSum);
							int index = tf.index(bri, i);
							float raw = a.get(in, index);
							float o = raw - max - logSum;
							a.add(out, index, o);
							//			check+=Math.exp(a.get(out, i));
						}
					}


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
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;

			ActivationSet a = as.a[b].get(t);
			ActivationSet e = errors.a[b].get(t);
			for(int bri = 0; bri<n0; ++bri) {
				for(LossGroup lg: lgs) {
					int iStart = lg.start;
					int iEnd = lg.end;
					boolean singleton = lg.isSingleton();

					if(singleton) {
						int i = iStart;
						int index = tf.index(bri, i);
						float od=e.get(out, index);
						float erg = a.get(out, index);
						float d =  erg * (1-erg);
						e.add(in, index, od*d);

					}else {
						float odSum=0;
						float sum = 0;
						double max=Double.NEGATIVE_INFINITY;
						int maxi=-1;
						for(int i=iStart; i<iEnd; ++i){
							double av = a.get(in, tf.index(bri, i));
							if(av>max){
								max=av;
								maxi=i;
							}
						}
						for(int i=iStart; i<iEnd; ++i){
							int index = tf.index(bri, i);
							odSum+=e.get(out, index);
							sum += Math.exp(a.get(in, index) - max);		
						}
						for(int i=iStart; i<iEnd; i++){
							int index = tf.index(bri, i);
							if(false && i==maxi){
								e.add(in, index, e.get(out, index) - odSum );
							}else{
								e.add(in, index, e.get(out, index) - odSum * (float)Math.exp(a.get(in, index) - max)/sum );
							}
						}

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
