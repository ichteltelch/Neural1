package org.siquod.ml.neural1.modules;

import java.util.Collections;
import java.util.HashSet;
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
import org.siquod.ml.neural1.modules.regularizer.Regularizer;

/**
 * This module performs a linear mapping from its input vector to its output vector
 * @author bb
 *
 */
public class Sparse implements InOutBiasModule{

	Interface in;
	Interface out;
	ParamBlock weights;
	ParamBlock bias;
	ParamBlock[] coordinateBias;
	int[] shift;
	int[][] connect;
	float filling;
	int[] posi;
	float[] coordinateActivations;
	boolean useBias;
	boolean[] useCoordinateBias;
	float biasInit=0;
	//	boolean useResidual;
	int dt;
	Regularizer reg;
	public Sparse(Sparse copyThis) {
		this.in = copyThis.in;
        this.out = copyThis.out;
        this.weights = copyThis.weights;
        this.bias = copyThis.bias;
        if(copyThis.coordinateBias!=null){
            this.coordinateBias=copyThis.coordinateBias.clone();
        }
        this.shift=copyThis.shift!=null?shift.clone():null;
        this.connect=copyThis.connect;
        this.filling=copyThis.filling;
        this.posi=copyThis.posi!=null?copyThis.posi.clone():null;
        this.coordinateActivations=copyThis.coordinateActivations!=null?copyThis.coordinateActivations.clone():null;
        this.useBias=copyThis.useBias;
        this.useCoordinateBias=copyThis.useCoordinateBias.clone();
        this.biasInit=copyThis.biasInit;
        this.reg=copyThis.reg;
        this.dt=copyThis.dt;     
	}
	@Override
	public Sparse copy() {
		return new Sparse(this);
	}
	public Sparse(float filling){
		this(filling, true, 0, null);
	}
	public Sparse(float filling, boolean useBias){
		this(filling, useBias, 0, null);
	}
	public Sparse(float filling, int dt, int [] shift, boolean... useCoordinateBias){
		this(filling, true, dt, shift, useCoordinateBias);
	}
	public Sparse(float filling, boolean useBias, int dt, int [] shift, boolean... useCoordinateBias){
		if(filling>0.9f)
			filling=0.9f;
		this.useBias=useBias;
		this.filling=filling;
		this.dt=dt;
		this.shift=shift==null?null:shift.clone();
		posi=shift==null?null:new int[shift.length];
		this.useCoordinateBias=new boolean[useCoordinateBias.length];
		System.arraycopy(useCoordinateBias, 0, this.useCoordinateBias, 0, useCoordinateBias.length);
		this.coordinateBias=new ParamBlock[useCoordinateBias.length];

	}
	public Sparse biasInit(float d) {biasInit=d; return this;}
	public void setCoordinateBiases(float[] cb) {
		coordinateActivations=cb;
	}
	public Sparse regularizer(Regularizer r){
		reg=r;
		return this;
	}
	//	public FullyConnected useResidual(){
	//		useResidual=true;
	//		return this;
	//	}
	//	public FullyConnected useResidual(boolean b){
	//		useResidual=b;
	//		return this;
	//	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		//		if(useResidual && in.count!=out.count){
		//			System.err.println("Canot make this a residual layer: input and output have different sizes");
		//			useResidual=false;
		//		}
	}

	@Override
	public void allocate(ParamAllocator ia) {
		int incount = in.channels();
		int outcount = out.channels();
		int fincount = (int)(incount*filling);
		connect=new int[outcount][fincount];
		int size = fincount * outcount;
		weights = ia.allocate(new ParamBlock("weights", size));
		if(useBias)
			bias = ia.allocate(new ParamBlock("bias", outcount));
		for(int i=0; i<useCoordinateBias.length; ++i) {
			if(useCoordinateBias[i])
				coordinateBias[i]=ia.allocate(new ParamBlock("coordinateBias"+i, outcount));
		}
		for(int[] sp: connect) {
			mid:for(int i=0; i<fincount; ++i) {
				int ns=(int) (Math.random()*incount);
				for(int j=0; j<i; ++j) {
					if(sp[j]==ns) {
						--i;
						continue mid;
					}
				}
				sp[i]=ns;
			}

		}
	}

	@Override
	public void share(ParamBlocks ps) {
		biasInit=Float.NaN;
		int size = in.channels() * out.channels();
		weights = ps.get("weights", size);
		if(useBias){
			bias = ps.get("bias", out.channels());
		}
		for(int i=0; i<useCoordinateBias.length; ++i) {
			if(useCoordinateBias[i])
				coordinateBias[i]=ps.get("coordinateBias"+i, out.channels());
		}
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("FullyConnected");
		ret.add(weights);
		if(useBias)
			ret.add(bias);
		for(int i=0; i<useCoordinateBias.length; ++i) {
			if(useCoordinateBias[i])
				ret.add(coordinateBias[i]);
		}
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(inst==null) {
			if(shift!=null)
				throw new IllegalArgumentException("This "+getClass().getName()+" module must be inside a convolution");
			int incount = in.count;
			int fincount = (int)(incount*filling);

			int outcount = out.count;
			for(ActivationSeq a: as) {
				if(a==null) continue;
				ActivationSet oa = a.get(t);
				ActivationSet ia = a.get(t+dt);
				if(ia!=null){
					for(int o=outcount-1; o>=0; --o){
						float sum=0;
						for(int i=fincount-1; i>=0; --i){
							float input = ia.get(in, connect[o][i]);
							if(input!=0)
								sum+= input * params.get(weights, o*fincount+i);
						}
						oa.add(out, o, sum);
					}
				}
				//		if(useResidual){
				//			for(int o=out.count-1; o>=0; --o){
				//				oa.add(out, o, ia.get(in, o));
				//			}
				//		}
				if(useBias){
					for(int o=outcount-1; o>=0; --o){
						oa.add(out, o, params.get(bias, o));
					}
				}
				for(int i=0; i<coordinateBias.length; ++i) {
					ParamBlock bias=coordinateBias[i];
					if(bias!=null) {
						float activation = coordinateActivations[i];
						for(int o=outcount-1; o>=0; --o){
							oa.add(out, o, activation*params.get(bias, o));
						}
					}
				}
			}		
		}else {
			if(shift==null)
				Module.copy(inst, posi);
			else
				Module.add(inst, shift, posi);
			int[] poso=inst;
			int incount = in.channels();
			int outcount = out.channels();
			int fincount = (int)(incount*filling);

			for(ActivationSeq a: as) {
				if(a==null) continue;
				ActivationSet oa = a.get(t);
				ActivationSet ia = a.get(t+dt);
				if(ia!=null){
					for(int o=outcount-1; o>=0; --o){
						float sum=0;
						for(int i=fincount-1; i>=0; --i){
							float input = ia.get(in, posi, connect[o][i]);
							if(input!=0)
								sum += input * params.get(weights, o*fincount+i);
						}
						oa.add(out, poso, o, sum);
					}
				}
				//		if(useResidual){
				//			for(int o=out.count-1; o>=0; --o){
				//				oa.add(out, o, ia.get(in, o));
				//			}
				//		}
				if(useBias){
					for(int o=outcount-1; o>=0; --o){
						oa.add(out, poso, o, params.get(bias, o));
					}
				}
				for(int i=0; i<coordinateBias.length; ++i) {
					ParamBlock bias=coordinateBias[i];
					if(bias!=null) {
						float activation = coordinateActivations[i];
						for(int o=outcount-1; o>=0; --o){
							oa.add(out, poso, o, activation*params.get(bias, o));
						}
					}
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {	
		if(dontBackprop.contains(phase))
			return;
		if(inst==null) {
			int incount = in.count;
			int fincount = (int)(incount*filling);

			int outcount = out.count;
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet inErr = errors.a[b].get(t+dt);
				if(inErr==null)
					continue;
				ActivationSet outErr = errors.a[b].get(t);
				if(outErr==null)
					continue;
				for(int o=outcount-1; o>=0; --o){
					float dist=outErr.get(out, o);
					if(dist==0) continue;
					for(int i=fincount-1; i>=0; --i){
						float summand=dist*params.get(weights, o*fincount+i);
						inErr.add(in, connect[o][i], summand);
					}
				}
			}
		}else {
			Module.add(inst, shift, posi);
			int[] poso=inst;
			int incount = in.channels();
			int fincount = (int)(incount*filling);

			int outcount = out.channels();
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet inErr = errors.a[b].get(t+dt);
				if(inErr==null)
					continue;
				ActivationSet outErr = errors.a[b].get(t);
				if(outErr==null)
					continue;
				for(int o=outcount-1; o>=0; --o){
					float dist = outErr.get(out, poso, o);
					if(dist==0)continue;
					for(int i=fincount-1; i>=0; --i){
						float summand = dist*params.get(weights, o*fincount+i);
						inErr.add(in, posi, connect[o][i], summand);
					}
				}
			}
		}
	}



	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, 
			ActivationBatch errors, ParamSet gradients, int t, int[] inst) {
		if(inst==null) {
			int incount = in.count;
			int fincount = (int)(incount*filling);

			int outcount = out.count;
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet input = as.a[b].get(t+dt);
				ActivationSet outErr = errors.a[b].get(t);
				if(input!=null){
					if(weights.shouldLearn(phase)){
						for(int i=fincount-1; i>=0; --i){
							for(int o=outcount-1; o>=0; --o){
								gradients.add(weights, o*fincount+i, outErr.get(out, o) * input.get(in, connect[o][i]));
							}
						}
					}
				}
				if(useBias){
					if(bias.shouldLearn(phase)){
						for(int o=outcount-1; o>=0; --o){
							gradients.add(bias, o, outErr.get(out, o));
						}
					}			
				}
				for(int i=0; i<coordinateBias.length; ++i) {
					ParamBlock bias=coordinateBias[i];
					if(bias!=null) {
						float activation = coordinateActivations[i];
						for(int o=outcount-1; o>=0; --o){
							gradients.add(bias, o, activation*outErr.get(out, o));
						}
					}
				}
			}
		}else{
			Module.add(inst, shift, posi);
			int[] poso=inst;
			int incount = in.channels();
			int fincount = (int)(incount*filling);

			int outcount = out.channels();
			for(int b=0; b<as.length; ++b) {
				if(errors.a[b]==null) continue;
				ActivationSet input = as.a[b].get(t+dt);
				ActivationSet outErr = errors.a[b].get(t);
				if(input!=null){
					if(weights.shouldLearn(phase)){
						for(int i=fincount-1; i>=0; --i){
							for(int o=outcount-1; o>=0; --o){
								gradients.add(weights, o*fincount+i, outErr.get(out, poso, o) * input.get(in, posi, connect[o][i]));
							}
						}
					}
				}
				if(useBias){
					if(bias.shouldLearn(phase)){
						for(int o=outcount-1; o>=0; --o){
							gradients.add(bias, o, outErr.get(out, poso, o));
						}
					}			
				}
				for(int i=0; i<coordinateBias.length; ++i) {
					ParamBlock bias=coordinateBias[i];
					if(bias!=null) {
						float activation = coordinateActivations[i];
						for(int o=outcount-1; o>=0; --o){
							gradients.add(bias, o, activation*outErr.get(out, poso, o));
						}
					}
				}
			}
		}
	}
	@Override
	public void regularize(String phase, ParamSet params, ParamSet gradients, float globReg) {
		if(reg!=null && weights.shouldLearn(phase)){
			int incount = in.channels();
			int outcount = out.channels();
			int fincount = (int)(incount*filling);
			reg.regularize(params, gradients, weights.start, 1, fincount, fincount, outcount, globReg);
		}
	}
	//	//	@Override
	//	public void declareDependencies(Dependencies d) {
	//		d.declare(new InputDependency(in, this, dt));
	//		d.declare(new OutputDependency(this, out));
	//	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}
	//	@Override
	public boolean wouldBackprop(String phase) {
		return !dontBackprop.contains(phase);
	}
	HashSet<String> dontBackprop=new HashSet<>();

	public Sparse dontBackprop(String phase){
		dontBackprop.add(phase);
		return this;
	}
	public String showParams(ParamSet ps) {
		StringBuilder ret=new StringBuilder();
		ret.append("matrix:\n");
		for(int i=0, w=0; i<in.count; ++i){
			for(int o=0; o<out.count; ++o, ++w){
				ret.append(ps.get(weights, w)).append(", \t");
			}
			ret.append("\n");
		}
		ret.append("bias:\n");
		if(useBias)
			for(int o=0; o<out.count; ++o){
				ret.append(ps.get(bias, o)).append(", \t");
			}

		return ret.toString();
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
		return shift;
	}
	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
	@Override
	public void initParams(ParamSet p) {
		if(bias!=null && !Double.isNaN(biasInit))
			for(int i=0; i<bias.count; ++i)
				p.set(bias, i, biasInit);
	}
	public ParamBlock getBias() {
		return bias;
	}
	public Sparse dt(int d){
		dt=d;
		return this;
	}
	public Sparse shift(int... shift){
		this.shift=shift==null?null:shift.clone();
		posi=shift==null?null:new int[shift.length];
		return this;
	}
} 
