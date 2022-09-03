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
import org.siquod.ml.neural1.ParamBlock;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;
import org.siquod.ml.neural1.neurons.ParameterizedNeuron;

/**
 * This module applies the same nonlinearity to all its input elements
 * @author bb
 *
 */
public class ParameterizedNonlin implements InOutModule{
	ParameterizedNeuron n;
	Interface in;
	Interface out;
	ParamBlock alpha;
	float min, max, regL1, regL2;
	TensorFormat tf;

	public ParameterizedNonlin(ParameterizedNeuron n, float min, float max, float regL1, float regL2){
		this.n=n;
		this.min = min;
		this.max = max;
		this.regL1 = regL1;
		this.regL2 = regL2;

	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out", in.count);
		if(out.tf==null)
			out.tf=in.tf;
		if(!in.tf.equals(out.tf))
			throw new IllegalArgumentException("input and output layer must be of the same size");
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
		alpha=ia.allocate(new ParamBlock("alpha", in.channels()));
	}

	@Override
	public void share(ParamBlocks ps) {
		return;
	}

	@Override
	public ParamBlocks getParamBlocks() {
		return null;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] pos) {
		if(pos==null) {
			int incount = tf.channels();
			int n0 = tf.dims[0];
			for(ActivationSeq b: as) {
				if(b==null) continue;

				ActivationSet a=b.get(t);
				for(int i=0; i<incount; ++i) {
					float alpha = params.get(this.alpha, i);
					float bounded = Math.max(Math.min(alpha, max), min);
					for(int bri = 0; bri < n0; ++bri) {
						int index = tf.index(bri, i);
						float inActivation = a.get(in, index);
						float outActivation = n.f(inActivation, bounded);
						a.add(out, index, outActivation);
					}
				}
			}
		}else {
			int incount = in.channels();
			for(ActivationSeq b: as) {
				if(b==null) continue;

				ActivationSet a=b.get(t);
				for(int i=0; i<incount; ++i) {
					float alpha = params.get(this.alpha, i);
					float bounded = Math.max(Math.min(alpha, max), min);
					int index = tf.index(pos, i);
					float inActivation = a.get(in, index);
					float outActivation = n.f(inActivation, bounded);
					a.add(out, index, outActivation);
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] pos) {
		if(pos==null) {
			int incount = in.count;
			int n0 = tf.dims[0];
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int i=0; i<incount; ++i) {
					float alpha = params.get(this.alpha, i);
					float bounded = Math.max(Math.min(alpha, max), min);
					for(int bri = 0; bri < n0; ++bri) {
						int index = tf.index(bri, i);
						float inActivation = a.get(in, index);
						e.add(in, index, e.get(out, index) * n.dfdx(inActivation, bounded));
					}
				}			
			}
		}else {
			int incount = in.channels();
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int i=0; i<incount; ++i) {
					float alpha = params.get(this.alpha, i);
					float bounded = Math.max(Math.min(alpha, max), min);
					int index = tf.index(pos, i);
					float inActivation = a.get(in, index);
					e.add(in, index, e.get(out, index) * n.dfdx(inActivation, bounded));
				}
			}
		}
	}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients, int t, int[] pos) {
		// o = a * f(i) + (1-a)*i
		// do
		// da = do * (f(i) - i)
		if(pos==null) {
			int incount = in.count;
			int n0 = tf.dims[0];
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int i=0; i<incount; ++i) {
					float alpha = params.get(this.alpha, i);
					float bounded = Math.max(Math.min(alpha, max), min);
					if(alpha>= min && alpha <= max) {
						float dBounded = 0;
						for(int bri = 0; bri < n0; ++bri) {
							int index = tf.index(bri, i);
							float linActivation = a.get(in, index);
							dBounded += e.get(out, index) * n.dfda(linActivation, bounded);
						}
						gradients.add(this.alpha, i, dBounded);
					}
				}			
			}
		}else {
			int incount = in.channels();
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet a=as.a[b].get(t);
				ActivationSet e=errors.a[b].get(t);
				for(int i=0; i<incount; ++i) {
					float alpha = params.get(this.alpha, i);
					float bounded = Math.max(Math.min(alpha, max), min);
					if(alpha>= min && alpha <= max) {
						float dBounded = 0;
						int index = tf.index(pos, i);
						float linActivation = a.get(in, index);
						dBounded += e.get(out, index) * n.dfda(linActivation, bounded);
						gradients.add(this.alpha, i, dBounded);
					}
				}
			}
		}
	}
	@Override
	public void regularize(String phase, ParamSet params, ParamSet gradients, float globReg) {
		if((regL1!=0 || regL2!=0) && alpha.shouldLearn(phase)) {
			int inCount = tf.channels();
			for(int i=0; i<inCount; ++i) {
				float a = params.get(alpha, i);
				float grad;
				if(a<=min) {
					float diff = min - a;
					grad = -(regL1 + regL2 * 2 * diff);
				}else if(a>=max) {
					float diff = a - max;
					grad = regL1 + regL2 * 2 * diff;
				}else {
					continue;
				}
				gradients.add(alpha, i, grad);
			}
		}
	}

	//	//	@Override
	//	public void declareDependencies(Dependencies d) {
	//		d.declare(new InputDependency(in, this, 0));
	//		d.declare(new OutputDependency(this, out));
	//	}
	@Override
	public void initParams(ParamSet p) {
		float v = (min + max) * 0.5f;
		for(int i=0; i<alpha.count; ++i)
			p.set(alpha, i, v);
	}
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
