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

public class Conv2D implements InOutBiasModule{

	Interface in, out;
	TensorFormat inf, outf;
	ParamBlock weights;
	ParamBlock bias;
	ParamBlock xBias;
	ParamBlock yBias;
	Regularizer reg;
	boolean useBias;
	boolean useCoordinateBias;

	int inChannels, outChannels;
	int inWx, outWx;
	int inWy, outWy;
	int fwx, fwy;
	int frx, fry;
	int dt;
	public Conv2D(int fr){
		this(fr, fr);
	}
	public Conv2D(int frx, int fry){
		this.frx=frx;
		this.fry=fry;
		fwx=frx*2+1;
		fwy=fry*2+1;
	}
	public Conv2D dt(int d){
		dt=d;
		return this;
	}
	float xBiasShift, yBiasShift;
	float xBiasScale, yBiasScale;
	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		inf=in.tf;
		outf=out.tf;
		if(inf==null)
			throw new IllegalArgumentException("input interface must specify a tensor format");
		if(inf.dims.length!=3)
			throw new IllegalArgumentException("input layer must be a rank-3 tensor");
		if(outf==null){
			int channelSize = inf.dims[0]*inf.dims[1];
			int outChannels=out.count/channelSize;
			int remainder = out.count - outChannels*channelSize;
			if(remainder!=0)
				throw new IllegalArgumentException("Cannot infer tensor format: output layer size is not a multiple of channel size");
			out.tf = outf = new TensorFormat(inf.dims[0], inf.dims[1], outChannels);
		}else{
			if(outf.dims.length!=3)
				throw new IllegalArgumentException("output layer must be a rank-3 tensor");
			if(inf.dims[0]!=outf.dims[0])
				throw new IllegalArgumentException("Incompatible tensor format");
			if(inf.dims[1]!=outf.dims[1])
				throw new IllegalArgumentException("Incompatible tensor format");
		}
		inWx=inf.dims[0];
		inWy=inf.dims[1];
		inChannels=inf.dims[2];
		outWx=outf.dims[0];
		outWy=outf.dims[1];
		outChannels=outf.dims[2];
		if(useCoordinateBias){
			int minDim=Math.min(outWx, outWy);
			xBiasScale = minDim*2.0f/(outWx*outWx);
			yBiasScale = minDim*2.0f/(outWy*outWy);
			xBiasShift = -outWx/2;
			yBiasShift = -outWy/2;
		}
	}

	@Override
	public void allocate(ParamAllocator ia) {
		int size = inChannels * outChannels * frx * fry;
		weights = ia.allocate(new ParamBlock("weights", size));
		if(useBias)
			bias = ia.allocate(new ParamBlock("bias", outChannels));
		if(useCoordinateBias){
			xBias=ia.allocate(new ParamBlock("xBias", outChannels));
			yBias=ia.allocate(new ParamBlock("yBias", outChannels));
		}
	}

	@Override
	public void share(ParamBlocks ps) {
		int size = inChannels * outChannels * frx * fry;
		weights = ps.get("weights", size);
		if(useBias){
			bias = ps.get("bias", outChannels);
		}		
		if(useCoordinateBias){
			xBias = ps.get("xBias", outChannels);
			yBias = ps.get("yBias", outChannels);
		}		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("Conv2D");
		ret.add(weights);
		if(useBias)
			ret.add(bias);
		if(useCoordinateBias){
			ret.add(xBias);
			ret.add(yBias);
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
						for(int y = 0; y<inWy; ++y){
							float sum=0;
							for(int i=inChannels-1, wi=wo*inChannels; i>=0; --i, wi++){
								for(int dx = -frx, wdx=wi*fwx; dx<=frx; ++dx, wdx++){
									for(int dy = -fry, wdy=wdx*fwy; dy<=fry; ++dy, wdy++){
										float weight = params.get(weights, wdy);
										float input = inf.get(ia, in, x+dx, y+dy, i);
										sum+=input * weight;
									}
								}
							}
							outf.add(oa, out, x, y, o, sum);
						}
					}
				}
			if(useBias){
				for(int o=outChannels-1; o>=0; --o){
					float weight = params.get(bias, o);
					for(int x = 0; x<inWx; ++x){
						for(int y = 0; y<inWy; ++y){
							outf.add(oa, out, x, y, o, weight);
						}
					}	
					oa.add(out, o, params.get(bias, o));
				}
			}
			if(useCoordinateBias){
				for(int o=outChannels-1; o>=0; --o){
					float weight = params.get(xBias, o)*xBiasScale;
					for(int x = 0; x<inWx; ++x){
						float input = (x + xBiasShift)*weight;
						for(int y = 0; y<inWy; ++y){
							outf.add(oa, out, x, y, o, input);
						}
					}	
				}
				for(int o=outChannels-1; o>=0; --o){
					float weight = params.get(yBias, o)*yBiasScale;
					for(int y = 0; y<inWy; ++y){
						float input = (y + yBiasShift)*weight;
						for(int x = 0; x<inWx; ++x){
							outf.add(oa, out, x, y, o, input);
						}
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
					for(int y = 0; y<inWy; ++y){
						float oe = outf.get(outErr, out, x, y, o);
						for(int i=inChannels-1, wi=wo*inChannels; i>=0; --i, wi++){
							for(int dx = -frx, wdx=wi*fwx; dx<=frx; ++dx, wdx++){
								for(int dy = -fry, wdy=wdx*fwy; dy<=fry; ++dy, wdy++){
									float weight = params.get(weights, wdy);
									inf.add(inErr, in, x+dx, y+dy, i, oe * weight);
								}
							}
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
			if(input==null){
				return;
			}
			ActivationSet outErr = errors.a[b].get(t);
			if(weights.shouldLearn(phase)){
				for(int o=outChannels-1, wo=0; o>=0; --o, wo++){
					for(int i=inChannels-1, wi=wo*inChannels; i>=0; --i, wi++){
						for(int dx = -frx, wdx=wi*fwx; dx<=frx; ++dx, wdx++){
							for(int dy = -fry, wdy=wdx*fwy; dy<=fry; ++dy, wdy++){
								float grad=0;
								for(int x = 0; x<inWx; ++x){
									for(int y = 0; y<inWy; ++y){
										grad += outf.get(outErr, out, x, y, o) * inf.get(input, in, x+dx, y+dy, i);
									}
								}
								gradients.add(weights, wdy, grad);
							}
						}
					}
				}
			}
			if(useBias){
				if(bias.shouldLearn(phase)){
					for(int o=outChannels-1; o>=0; --o){
						float grad=0;
						for(int x = 0; x<inWx; ++x){
							for(int y = 0; y<inWy; ++y){
								grad += outf.get(outErr, out, x, y, o);
							}
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
							for(int y = 0; y<inWy; ++y){
								grad += outf.get(outErr, out, x, y, o)*inp;
							}
						}
						gradients.add(xBias, o, grad);
					}
				}	
				if(yBias.shouldLearn(phase)){
					for(int o=outChannels-1; o>=0; --o){
						float grad=0;
						for(int y = 0; y<inWy; ++y){
							float inp = (y + yBiasShift)*yBiasScale;
							for(int x = 0; x<inWx; ++x){
								grad += outf.get(outErr, out, x, y, o)*inp;
							}
						}
						gradients.add(yBias, o, grad);
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
	public Conv2D regularizer(Regularizer r){
		reg=r;
		return this;
	}
	@Override
	public ParamBlock getBias() {
		return bias;
	}
}
