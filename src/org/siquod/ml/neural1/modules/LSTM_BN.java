package org.siquod.ml.neural1.modules;

import java.util.Arrays;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlock;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;
import org.siquod.ml.neural1.neurons.Fermi;
import org.siquod.ml.neural1.neurons.Tanh;

public class LSTM_BN implements InOutModule{


	Interface c;
	GateLayer g1; //forget gate
	GateLayer g2=new GateLayer();
	GateLayer g3=new GateLayer();
	Nonlin s1=new Nonlin(Fermi.INST);
	Nonlin s2=new Nonlin(Fermi.INST);
	Nonlin s3=new Nonlin(Fermi.INST);
	Nonlin t1=new Nonlin(Tanh.INST);
	Nonlin t2=new Nonlin(Tanh.INST);
	InOutScaleBiasModule bnx;
	InOutScaleBiasModule bnh;
	InOutScaleBiasModule bno;
	Interface four_h;
	Interface four_x;
	Interface four;
	SplitLayer split;
	Interface s1o, s2o, s3o;
	Interface t1o, t2o;
	Interface s1i, s2i, s3i;
	Interface t1i, t2i;
	Interface hi;
	BackpropStopper trunc;
	InOutModule hMat, xMat;
	Interface in, out;
	Module[] exec;
	int dt=-1;
	int[] shift;
	public LSTM_BN(InOutModule hMat, InOutModule xMat){
		this(true, hMat, xMat);
	}
	/**
	 * The xMat module connects the input from the lower layer to the nonlinearities.
	 * The hMat module connects to past output of the LSTM cells to the nonlinearities.
	 * The dt and shift values of the hMat module determine in which direction time runs.
	 * @param hMat
	 * @param xMat
	 */
	public LSTM_BN(Boolean useReNorm, InOutModule hMat, InOutModule xMat){
		this.hMat=hMat;
		this.xMat=xMat;
		dt=hMat.dt();
		shift=hMat.shift();
		if(shift!=null)
			shift=shift.clone();
		if(useReNorm==null) {
			bnx=new LayerNorm(true);
			bnh=new LayerNorm(false);
			bno=new LayerNorm();
		}else if(useReNorm) {
			bnx=new BatchReNorm(true);
			bnh=new BatchReNorm(false);
			bno=new BatchReNorm();
		}else {
			bnx=new BatchNorm(true);
			bnh=new BatchNorm(false);
			bno=new BatchNorm();
		}
	}
	double forgetBiasInit=1;
	public LSTM_BN forgetBiasInit(double d) {
		forgetBiasInit=d;
		return this;
	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		if(hMat==null)
			hMat=new Dense();
		if(xMat==null)
			xMat=new Dense();
		g1=new GateLayer(dt, 0, shift, null);
		in=ia.get("in");
		out=ia.get("out");
		TensorFormat s = out.tf;
		int sc=s.count();
		c=ia.allocate(new Interface("c", s));
		four=ia.allocate(new Interface("4", s.withChannels(4*s.channels())));
		four_h=ia.allocate(new Interface("4h", s.withChannels(4*s.channels())));
		four_x=ia.allocate(new Interface("4x", s.withChannels(4*s.channels())));
		split=new SplitLayer()
				.output(0, t1i=new Interface("t1i", s))
				.output(sc, s1i=new Interface("s1i", s))
				.output(2*sc, s2i=new Interface("s2i", s))
				.output(3*sc, s3i=new Interface("s3i", s));
		s1o=ia.allocate(new Interface("s1o", s));
		s2o=ia.allocate(new Interface("s2o", s));
		s3o=ia.allocate(new Interface("s3o", s));
		t1o=ia.allocate(new Interface("t1o", s));
		t2o=ia.allocate(new Interface("t2o", s));
		Interface copy;
		Copy cl=null;
		if(!! true) {
			cl=new Copy(dt, shift);
			copy=ia.allocate(new Interface("outcopy", out.tf));
			cl.allocate(ia, "out", "outcopy");
			hMat.dt(0).shift(null);
		}else {
			copy=out;
		}

		trunc=new BackpropStopper(copy, "hi", 0, null);
		//		trunc=new BackpropStopper(out, "hi", dt, shift);
		trunc.allocate(ia);
		hi=trunc.out;
		hMat.allocate(ia, "hi", "4h");
		xMat.allocate(ia, "in", "4x");
		bnh.allocate(ia, "4h", "4");
		bnx.allocate(ia, "4x", "4");

		split.allocate(ia, "4");
		ia.allocate(s1i);
		ia.allocate(s2i);
		ia.allocate(s3i);
		ia.allocate(t1i);
		s1.allocate(ia, "s1i", "s1o");
		s2.allocate(ia, "s2i", "s2o");
		s3.allocate(ia, "s3i", "s3o");
		t1.allocate(ia, "t1i", "t1o");
		g1.allocate(ia, "c", "s1o", "c");
		g2.allocate(ia, "s2o", "t1o", "c");
		t2i=ia.allocate(new Interface("t2i", s));
		bno.allocate(ia, "c", "t2i");
		t2.allocate(ia, "t2i", "t2o");
		g3.allocate(ia, "s3o", "t2o", "out");
		// in -> xMat -> 4
		// 4 -> {s1i, s2i, t1i, s3i}
		// s1i -> s1 -> s1o
		// s2i -> s2 -> s2o
		// s3i -> s3 -> s3o
		// t1i -> t1 -> t1o
		// {c-, s1o} -> g1 -> c
		// {s2o, t1o} -> g2 -> c
		// c -> t2 -> t2o
		// {s3o, t2o} -> g3 -> out
		// out- -> hMat -> 4

		exec=new Module[] {
				cl,  
				trunc, 
				hMat, bnh,
				xMat, bnx,
				split,
				s1, s2, s3, t1,
				g1, g2, 
				bno, t2, 
				g3
		};
	}

	@Override
	public void allocate(ParamAllocator ia) {
		ia.push(null); xMat.allocate(ia); ia.pop();
		ia.push(null); bnx.allocate(ia); ia.pop();
		ia.push(null); bno.allocate(ia); ia.pop();
		ia.push(null); hMat.allocate(ia); ia.pop();
		ia.push(null); bnh.allocate(ia); ia.pop();
	}

	@Override
	public void share(ParamBlocks ps) {
		xMat.share(ps.get("xMat"));
		hMat.share(ps.get("hMat"));
		g1.share(ps.get("g1"));
		g2.share(ps.get("g2"));
		g3.share(ps.get("g3"));
		s1.share(ps.get("s1"));
		s2.share(ps.get("s2"));
		s3.share(ps.get("s3"));
		t1.share(ps.get("t1"));
		t2.share(ps.get("t2"));
		bnx.share(ps.get("bnx"));
		bnh.share(ps.get("bnh"));
		bno.share(ps.get("bno"));
		split.share(ps.get("split"));
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret=new ParamBlocks("LSTM");
		ret.add("xMat", xMat.getParamBlocks());
		ret.add("hMat", hMat.getParamBlocks());
		ret.add("bnx", bnx.getParamBlocks());
		ret.add("bnh", bnh.getParamBlocks());
		ret.add("bno", bno.getParamBlocks());
		ret.add("g1", g1.getParamBlocks());
		ret.add("g2", g2.getParamBlocks());
		ret.add("g3", g3.getParamBlocks());
		ret.add("s1", s1.getParamBlocks());
		ret.add("s2", s2.getParamBlocks());
		ret.add("s3", s3.getParamBlocks());
		ret.add("t1", t1.getParamBlocks());
		ret.add("t2", t2.getParamBlocks());
		ret.add("split", split.getParamBlocks());
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		//		for(int i=0; i<bnh.mult.count; ++i) {
		//			params.set(bnh.mult, i, 0);
		////			params.set(bnh.add, i, 0);
		//		}
		for(Module m: exec)
			m.forward(training, params, as, t, inst);

	}
	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int i=exec.length-1; i>=0; --i) {
			exec[i].backprop(phase, params, as, errors, t, inst);
		}

	}
	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		for(int i=exec.length-1; i>=0; --i)
			exec[i].gradients(phase, params, as, errors, gradients, t, inst);
	}
	@Override
	public List<Module> getSubmodules() {
		return Arrays.asList(exec);
	}

	////	@Override
	//	public void declareDependencies(Dependencies d) {
	////		hMat.declareDependencies(d);
	////		xMat.declareDependencies(d);
	//		g1.declareDependencies(d);
	//		g2.declareDependencies(d);
	//		g3.declareDependencies(d);
	//		s1.declareDependencies(d);
	//		s2.declareDependencies(d);
	//		s3.declareDependencies(d);
	//		t1.declareDependencies(d);
	//		t2.declareDependencies(d);
	//		split.declareDependencies(d);
	//	}

	@Override
	public void dontComputeInPhase(String phase) {
	}

	//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
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
	public Interface getIn() {
		return in;
	}

	@Override
	public Interface getOut() {
		return out;
	}
	@Override
	public void initParams(ParamSet p) {
		defaultInitParams(p);
		ParamBlock pb=bnx.getBias();
		int n = pb.count/4;
		for(int i=0; i<n; ++i) {
			p.set(pb, i, forgetBiasInit);
			//			p.set(bnx.add, n+i, 100);
			//			p.set(bnx.add, 3*n+i, 100);
		}
		//		for(int i=0; i<bnh.mult.count; ++i) {
		//			p.set(bnh.mult, i, p.get(bnh.mult, i)/16);
		//		}
		pb=bnx.getScale();
		for(int i=0; i<pb.count; ++i) {
			p.set(pb, i, 0);
		}
	}

}
