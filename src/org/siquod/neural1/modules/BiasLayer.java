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
import org.siquod.neural1.ParamBlock;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;

public class BiasLayer extends InOutCastLayer implements HasBias{
	ParamBlock bias;
	public BiasLayer(Interface in2) {
		super(in2);
		out=new Interface(in.count, in.tf);
		out.offset=in.offset;
	}
	public static InOutCastFactory factory(){
		return new InOutCastFactory() {
			@Override
			public InOutCastLayer produce(Interface in) {
				return new BiasLayer(in);
			}
		};
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		super.allocate(ia);
		if(out.tf==null)
			out.tf=in.tf;

	}
	@Override
	public void allocate(ParamAllocator ia) {
		bias=ia.allocate(new ParamBlock("bias", in.channels()));
	}
	@Override
	public void share(ParamBlocks ps) {
		bias=ps.get("bias", in.channels());
	}
	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("biasLayer");
		ret.add(bias);
		return ret;
	}
	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(inst!=null)
			throw new IllegalArgumentException("A "+getClass().getName()+" module must not be inside a convolution");
		int incount=in.channels();
		if(inst==null) {
			if(in.tf.rank>1) {
				int stride = in.tf.channelStride();
				for(ActivationSeq ab: as.a) {
					ActivationSet a=ab.get(t);
					for(int c=0; c<incount; ++c) {
						int o=stride*c;
						float b=params.get(bias, c);
						for(int i=0; i<stride; ++i) {
							a.add(in, o+i, b);
						}
					}
				}
			}else {
				for(ActivationSeq ab: as.a) {
					ActivationSet a=ab.get(t);
					for(int c=0; c<incount; ++c) {
						float b=params.get(bias, c);
						a.add(in, c, b);
					}
				}
			}
		}else{
			int stride = in.tf.channelStride();
			for(ActivationSeq ab: as.a) {
				ActivationSet a=ab.get(t);
				int i = in.tf.index(inst, 0);
				for(int c=0; c<incount; ++c) {
					int o=stride*c;
					float b=params.get(bias, c);
					a.add(in, o+i, b);
				}
			}			
			
		}
	}
	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
	}
	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		int incount=in.channels();
		if(inst==null) {
			if(in.tf.rank>1) {
				int stride = in.tf.channelStride();
				for(int bi=0; bi<as.length; ++bi) {
					ActivationSeq eb=errors.a[bi];
					ActivationSet e=eb.get(t);
					for(int c=0; c<incount; ++c) {
						int o=stride*c;
						float te=0;
						for(int i=0; i<stride; ++i) {
							te += e.get(in, o+i);
						}
						gradients.add(c, te);
					}
				}
			}else {
				for(int bi=0; bi<as.length; ++bi) {
					ActivationSeq eb=errors.a[bi];
					ActivationSet e=eb.get(t);
					for(int c=0; c<incount; ++c) {
						float te;
						te = e.get(in, c);
						gradients.add(c, te);
					}
				}
			}
		}else{
			int stride = in.tf.channelStride();
			for(int bi=0; bi<as.length; ++bi) {
				ActivationSeq eb=errors.a[bi];
				ActivationSet e=eb.get(t);
				int i = in.tf.index(inst, 0);
				for(int c=0; c<incount; ++c) {
					int o=stride*c;
					float te;
					te = e.get(in, o+i);
					gradients.add(c, te);
				}
			}			
			
		}	}

	@Override
	public void dontComputeInPhase(String phase) {
	}
	//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
	}

	@Override
	public void initParams(ParamSet p) {
		//TODO
	}

	@Override
	public void initializeRun(ActivationBatch as, boolean training) {
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
	@Override
	public ParamBlock getBias() {
		return bias;
	}
}
