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
import org.siquod.neural1.TensorFormat;
import org.siquod.neural1.modules.regularizer.Regularizer;

public class Conv1D implements InOutBiasModule{

	Interface in, out;
	TensorFormat inf, outf;
	ParamBlock weights;
	ParamBlock bias;
	ParamBlock xBias;
	Regularizer reg;
	boolean useBias=true;
	boolean useCoordinateBias=true;

	int inChannels, outChannels;
	int inWx, outWx;
	int fwx;
	int frx;
	int dt;
	public Conv1D(int fr){
		this.frx=fr;
		fwx=frx*2+1;
	}
	public Conv1D dt(int d){
		dt=d;
		return this;
	}
	float xBiasShift;
	float xBiasScale;
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
		if(outf==null){
			int channelSize = inf.dims[0];
			int outChannels=out.count/channelSize;
			int remainder = out.count - outChannels*channelSize;
			if(remainder!=0)
				throw new IllegalArgumentException("Cannot infer tensor format: output layer size is not a multiple of channel size");
			out.tf = outf = new TensorFormat(inf.dims[0], outChannels);
		}else{
			if(outf.dims.length!=2)
				throw new IllegalArgumentException("output layer must be a rank-2 tensor");
			if(inf.dims[0]!=outf.dims[0])
				throw new IllegalArgumentException("Incompatible tensor format");
		}
		inWx=inf.dims[0];
		inChannels=inf.dims[1];
		outWx=outf.dims[0];
		outChannels=outf.dims[1];
		if(useCoordinateBias){
			xBiasScale = 2.0f/outWx;
			xBiasShift = -outWx/2;
		}

	}

	@Override
	public void allocate(ParamAllocator ia) {
		int size = inChannels * outChannels * fwx;
		weights = ia.allocate(new ParamBlock("weights", size));
		if(useBias)
			bias = ia.allocate(new ParamBlock("bias", outChannels));
		if(useCoordinateBias){
			xBias=ia.allocate(new ParamBlock("xBias", outChannels));
		}
	}

	@Override
	public void share(ParamBlocks ps) {
		int size = inChannels * outChannels * fwx;
		weights = ps.get("weights", size);
		if(useBias){
			bias = ps.get("bias", outChannels);
		}		
		if(useCoordinateBias){
			xBias = ps.get("xBias", outChannels);
		}		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("Conv1D");
		ret.add(weights);
		if(useBias)
			ret.add(bias);
		if(useCoordinateBias){
			ret.add(xBias);
		}
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {

		if(inst!=null)
			throw new IllegalArgumentException("A "+getClass().getName()+" module must not be inside a convolution");
		for(ActivationSeq b: as) {
			if(b==null) continue;
			ActivationSet oa = b.get(t);
			ActivationSet ia = b.get(t+dt);
			if(ia!=null)
				for(int o=outChannels-1, wo=0; o>=0; --o, wo++){
					for(int x = 0; x<inWx; ++x){
						float sum=0;
						for(int i=inChannels-1, wi=wo*inChannels; i>=0; --i, wi++){
							for(int dx = -frx, wdx=wi*fwx; dx<=frx; ++dx, wdx++){
								float weight = params.get(weights, wdx);
								float input = inf.get(ia, in, x+dx, i);
								sum+=input * weight;

							}
						}
						outf.add(oa, out, x, o, sum);
					}

				}
			if(useBias){
				for(int o=outChannels-1; o>=0; --o){
					float weight = params.get(bias, o);
					for(int x = 0; x<inWx; ++x){
						outf.add(oa, out, x, o, weight);

					}	
					oa.add(out, o, params.get(bias, o));
				}
			}
			if(useCoordinateBias){
				for(int o=outChannels-1; o>=0; --o){
					float weight = params.get(xBias, o)*xBiasScale;
					for(int x = 0; x<inWx; ++x){
						float input = (x + xBiasShift)*weight;
						outf.add(oa, out, x, o, input);
					}	
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet inErr = errors.a[b].get(t+dt);
			if(inErr==null)
				return;
			ActivationSet outErr = errors.a[b].get(t);
			for(int o=outChannels-1, wo=0; o>=0; --o, wo++){
				for(int x = 0; x<inWx; ++x){
					float oe = outf.get(outErr, out, x, o);
					for(int i=inChannels-1, wi=wo*inChannels; i>=0; --i, wi++){
						for(int dx = -frx, wdx=wi*fwx; dx<=frx; ++dx, wdx++){
							float weight = params.get(weights, wdx);
							inf.add(inErr, in, x+dx, i, oe * weight);
						}
					}
				}

			}
		}
	}
	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, 
			ActivationBatch errors, ParamSet gradients, int t, int[] inst) {
		for(int b=0; b<as.length; ++b) {
			if(errors.a[b]==null) continue;
			ActivationSet input = as.a[b].get(t);
			ActivationSet outErr = errors.a[b].get(t);
			if(weights.shouldLearn(phase)){
				for(int o=outChannels-1, wo=0; o>=0; --o, wo++){
					for(int i=inChannels-1, wi=wo*inChannels; i>=0; --i, wi++){
						for(int dx = -frx, wdx=wi*fwx; dx<=frx; ++dx, wdx++){
							float grad=0;
							for(int x = 0; x<inWx; ++x){
								grad += outf.get(outErr, out, x, o) * inf.get(input, in, x+dx, i);
							}
							gradients.add(weights, wdx, grad);
						}
					}
				}
			}
			if(useBias){
				if(bias.shouldLearn(phase)){
					for(int o=outChannels-1; o>=0; --o){
						float grad=0;
						for(int x = 0; x<inWx; ++x){
							grad += outf.get(outErr, out, x, o);
						}
						gradients.add(bias, o, grad);
					}
				}			
			}
			if(useCoordinateBias){
				if(xBias.shouldLearn(phase)){
					for(int o=outChannels-1; o>=0; --o){
						float grad=0;
						for(int x = 0; x<inWx; ++x){
							float inp = (x + xBiasShift)*xBiasScale;
							grad += outf.get(outErr, out, x, o)*inp;
						}
						gradients.add(xBias, o, grad);
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
	public void regularize(String phase, ParamSet params, ParamSet gradients,
			float globReg) {
		if(reg!=null && weights.shouldLearn(phase)){
			reg.regularize(params, gradients, weights.start, outChannels, 1, inChannels, outChannels, globReg);
		}
	}
	public Conv1D regularizer(Regularizer r){
		reg=r;
		return this;
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
		return dt;
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
