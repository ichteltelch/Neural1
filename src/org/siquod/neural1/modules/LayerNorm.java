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

public class LayerNorm implements InOutScaleBiasModule{

	private static final float epsilon = 1e-10f;

	Interface in, out;
	Interface stat;
	ParamBlock add, mult;
	private boolean hasAdd=true;

	public LayerNorm() {
		this(true);
	}
	public LayerNorm(boolean b) {
		hasAdd=b;
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
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		if(in.count!=out.count)
			throw new IllegalArgumentException("Input layer size must output layer size");
		stat=ia.allocate(new Interface("stat", in.tf.withChannels(2)));		
	}

	@Override
	public void allocate(ParamAllocator ia) {
		if(hasAdd) add = ia.allocate(new ParamBlock("add", in.channels()));
		mult = ia.allocate(new ParamBlock("mult", in.channels()));
	}

	@Override
	public void share(ParamBlocks ps) {
		if(hasAdd) add = ps.get("add", in.channels());
		mult = ps.get("mult", in.channels());		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("Conv1D");
		if(hasAdd) ret.add(add);
		ret.add(mult);
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		if(inst==null) {
			int incount = in.count;
			for(ActivationSeq b: as.a) {
				if(b==null)continue;
				ActivationSet a=b.get(t);
				if(a==null) continue;
				float mean=0;
				float var=0;
				for(int i=0; i<incount; ++i) {
					mean += a.get(in, i);
				}
				mean/=incount;
				for(int i=0; i<incount; ++i) {
					float d = a.get(in, i)-mean;
					var += d*d;
				}
				var/=incount;
				float sdev=(float) Math.sqrt(var+epsilon);
				float scal = 1/sdev;
				a.set(stat, 0, mean);
				a.set(stat, 1, sdev);
				for(int i=0; i<incount; ++i) {
					float ni = (a.get(in, i)-mean)*scal;
					float rs=(hasAdd?params.get(add, i):0) + params.get(mult, i)*ni;
					a.add(out, i, rs);
				}
			}
		}else { 
			int incount = in.channels();
			for(ActivationSeq b: as.a) {
				if(b==null)continue;
				ActivationSet a=b.get(t);
				if(a==null) continue;
				float mean=0;
				float var=0;
				for(int i=0; i<incount; ++i) {
					mean += a.get(in, inst, i);
				}
				mean/=incount;
				for(int i=0; i<incount; ++i) {
					float d = a.get(in, inst, i)-mean;
					var += d*d;
				}
				var/=incount;
				float sdev=(float) Math.sqrt(var+epsilon);
				float scal = 1/sdev;
				a.set(stat, inst, 0, mean);
				a.set(stat, inst, 1, sdev);
				for(int i=0; i<incount; ++i) {
					float ni = (a.get(in, inst, i)-mean)*scal;
					float rs=(hasAdd?params.get(add, i):0) + params.get(mult, i)*ni;
					a.add(out, inst, i, rs);
				}
			}
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		if(inst==null) {
			int incount = in.count;
			for(int b=0; b<as.length; ++b) {
				ActivationSeq asq=as.a[b];
				if(asq==null)continue;
				ActivationSeq esq=errors.a[b];
				if(esq==null)continue;
				ActivationSet a=asq.get(t);
				ActivationSet e=esq.get(t);
				if(a==null) continue;
				float mean=a.get(stat, 0);
				float sdev=a.get(stat, 1);
				float scal = 1/sdev;
				float scalErr=0;
				for(int i=0; i<incount; ++i) {
					float multVal = params.get(mult, i);
					float oe=e.get(out, i);
					oe*=multVal;
					float scalInp=a.get(in, i)-mean;
					scalErr += oe*scalInp;
				}
				float varErr = scalErr * -0.5f / (sdev*sdev*sdev);
				varErr/=incount;
				float meanErr=0;
				for(int i=0; i<incount; ++i) {
					float multVal = params.get(mult, i);
					float oe=e.get(out, i);
					oe*=multVal*scal;
					float scalInp=a.get(in, i)-mean;
					oe+=2*varErr*scalInp;
					meanErr -= oe;
				}	
				meanErr/=incount;
				for(int i=0; i<incount; ++i) {
					float multVal = params.get(mult, i);
					float oe=e.get(out, i);
					oe*=multVal*scal;
					float scalInp=a.get(in, i)-mean;
					oe+=2*varErr*scalInp;
					oe+=meanErr;
					e.add(in, i, oe);
				}			
			}
		}else { 
			int incount = in.channels();
			for(int b=0; b<as.length; ++b) {
				ActivationSeq asq=as.a[b];
				if(asq==null)continue;
				ActivationSeq esq=errors.a[b];
				if(esq==null)continue;
				ActivationSet a=asq.get(t);
				ActivationSet e=esq.get(t);
				if(a==null) continue;
				float mean=a.get(stat, inst, 0);
				float sdev=a.get(stat, inst, 1);
				float scal = 1/sdev;
				float scalErr=0;
				for(int i=0; i<incount; ++i) {
					float multVal = params.get(mult, i);
					float oe=e.get(out, inst, i);
					oe*=multVal;
					float scalInp=a.get(in, inst, i)-mean;
					scalErr += oe*scalInp;
				}
				float varErr = scalErr * -0.5f / (sdev*sdev*sdev);
				varErr/=incount;
				float meanErr=0;
				for(int i=0; i<incount; ++i) {
					float multVal = params.get(mult, i);
					float oe=e.get(out, inst, i);
					oe*=multVal*scal;
					float scalInp=a.get(in, inst, i)-mean;
					oe+=2*varErr*scalInp;
					meanErr -= oe;
				}	
				meanErr/=incount;
				for(int i=0; i<incount; ++i) {
					float multVal = params.get(mult, i);
					float oe=e.get(out, inst, i);
					oe*=multVal*scal;
					float scalInp=a.get(in, inst, i)-mean;
					oe+=2*varErr*scalInp;
					oe+=meanErr;
					e.add(in, i, oe);
				}			
			}
		}

	}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		if(inst==null) {
			int incount = in.count;
			for(int b=0; b<as.length; ++b) {
				ActivationSeq asq=as.a[b];
				if(asq==null)continue;
				ActivationSeq esq=errors.a[b];
				if(esq==null)continue;
				ActivationSet a=asq.get(t);
				ActivationSet e=esq.get(t);
				if(a==null) continue;
				float mean=a.get(stat, 0);
				float sdev=a.get(stat, 1);
				float scal = 1/sdev;
				for(int i=0; i<incount; ++i) {
					float multInp=(a.get(in, i)-mean)*scal;
					float oe=e.get(out, i);
					gradients.add(mult, i, oe*multInp);
					if(hasAdd)
						gradients.add(add, i, oe);
				}
			}
		}else {
			int incount = in.channels();
			for(int b=0; b<as.length; ++b) {
				ActivationSeq asq=as.a[b];
				if(asq==null)continue;
				ActivationSeq esq=errors.a[b];
				if(esq==null)continue;
				ActivationSet a=asq.get(t);
				ActivationSet e=esq.get(t);
				if(a==null) continue;
				float mean=a.get(stat, inst, 0);
				float sdev=a.get(stat, inst, 1);
				float scal = 1/sdev;
				for(int i=0; i<incount; ++i) {
					float multInp=(a.get(in, inst, i)-mean)*scal;
					float oe=e.get(out, inst, i);
					gradients.add(mult, i, oe*multInp);
					if(hasAdd)
						gradients.add(add, i, oe);
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
	public ParamBlock getBias() {
		return add;
	}

	@Override
	public ParamBlock getScale() {
		return mult;
	}
	@Override
	public void initParams(ParamSet p) {
		for(int i=0; i<in.count; ++i) {
			if(hasAdd)
				p.set(add, i, 0);
			p.set(mult, i, 1);
		}
	}

}
