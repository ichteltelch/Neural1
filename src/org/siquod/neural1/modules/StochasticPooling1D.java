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

public class StochasticPooling1D implements InOutModule{

	private Interface in;
	private Interface out;
	private TensorFormat inf;
	private TensorFormat outf;
	int factor=2;
	float[] select;
	{
		select=new float[factor];
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		inf=in.tf;
		outf=out.tf;
		if(inf==null)
			throw new IllegalArgumentException("input interface must specify a tensor format");
		if(inf.dims.length!=2)
			throw new IllegalArgumentException("input layer must be a rank-2 tensor");
		if(outf==null)
			throw new IllegalArgumentException("output interface must specify a tensor format");
		if(outf.dims.length!=2)
			throw new IllegalArgumentException("output layer must be a rank-2 tensor");
		if(inf.dims[1] != outf.dims[1])
			throw new IllegalArgumentException("input and output layer must have the same number of channels");
		ia.allocateDecision(outf.count());
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
		if(inst!=null)
			throw new IllegalThreadStateException("A "+getClass().getName()+" module must not be inside a convolution");
		if(training==ForwardPhase.TRAINING) {
			for(ActivationSeq b: as.a) {
				ActivationSet ia=b.get(t);
				ActivationSet oa=ia;
				for(int i=outf.dims[1]-1, odec=0; i>=0; --i) {
					for(int j=outf.dims[0]-1; j>=0; --j, odec++) {
						int offset=j*factor;
						float max=Float.NEGATIVE_INFINITY;

						for(int k=0; k<factor; ++k) {
							float s = select[k]=inf.get(ia, in, offset+k, i);
							if(s>max)
								max=s;
						}
						float sum=0;
						for(int k=0; k<factor; ++k){
							float av = select[k];			
							//							sum += select[k] = Math.exp(av-max);
							sum += select[k] = Math.max(av, 0);
						}
						int choice=0;
						if(sum==0) {
							choice=(int) (Math.random()*factor);
						}else {
							float rand=(float) (Math.random()*sum);
							while(rand>0 && choice<factor) {
								rand-=select[choice++];
							}
							if(choice>=factor)
								choice=factor;
						}
						oa.setDecision(odec, choice);
						outf.add(oa, out, j, i, inf.get(ia, in, offset+choice, i));
					}
				}
			}
		}else {
			for(ActivationSeq b: as.a) {
				ActivationSet ia=b.get(t);
				ActivationSet oa=ia;
				for(int i=outf.dims[1]-1; i>=0; --i) {
					for(int j=outf.dims[0]-1; j>=0; --j) {
						int offset=j*factor;
						float max=Float.NEGATIVE_INFINITY;

						for(int k=0; k<factor; ++k) {
							float s = select[k]=inf.get(ia, in, offset+k, i);
							if(s>max)
								max=s;
						}
						float weight=0, sum=0;
						for(int k=0; k<factor; ++k){
							float av = select[k];
							float s = Math.max(av, 0);
							//							float s = Math.exp(av-max);
							weight += s;
							sum += s*av;
						}
						if(weight>0) {
							outf.add(oa, out, j, i, sum/weight);
						}else {
							
						}
					}
				}
			}	
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int b=0; b<as.length; ++b) {
			ActivationSeq ba=as.a[b];
			ActivationSeq be=errors.a[b];
			ActivationSet ie=be.get(t);
			ActivationSet oe=ie;
			ActivationSet oa=ba.get(t);
			for(int i=outf.dims[1]-1, odec=0; i>=0; --i) {
				for(int j=outf.dims[0]-1; j>=0; --j, odec++) {
					int offset=j*factor;
					int choice=oa.getDecision(odec);
					inf.add(ie, in, offset+choice, i, outf.get(oe, out, j, i));
				}
			}
		}

	}

	@Override
	public void dontComputeInPhase(String phase) {
	}

	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
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

}
