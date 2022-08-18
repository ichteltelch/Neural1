//package org.siquod.neural1.modules;
//
//import java.util.Arrays;
//import java.util.List;
//
//import org.siquod.neural1.ActivationBatch;
//import org.siquod.neural1.ForwardPhase;
//import org.siquod.neural1.Interface;
//import org.siquod.neural1.InterfaceAllocator;
//import org.siquod.neural1.Module;
//import org.siquod.neural1.ParamAllocator;
//import org.siquod.neural1.ParamBlock;
//import org.siquod.neural1.ParamBlocks;
//import org.siquod.neural1.ParamSet;
//import org.siquod.neural1.TensorFormat;
//import org.siquod.neural1.data.PolyInteraction;
//import org.siquod.neural1.neurons.Fermi;
//import org.siquod.neural1.neurons.Tanh;
//
//public class Enrichment implements InOutModule{
//Interface in, out;
//	
//	Module[] exec;
//	Interface[] projections;
//	Interface[] beforeJoining;
//	Dense[] projectors;
//	PolyInteractionModule[] interactions;
//	int[] offsets;
//	int[] dims;
//	Copy copy=new Copy();
//	JoinLayer join;
//	int inputWidth;
//	
//	
//	
//	/**
//	 * The xMat module connects the input from the lower layer to the nonlinearities.
//	 * The hMat module connects to past output of the LSTM cells to the nonlinearities.
//	 * The dt and shift values of the hMat module determine in which direction time runs.
//	 * @param hMat
//	 * @param xMat
//	 */
//	public Enrichment(int... dimensions){
//		int count = 0;
//		for(int dim: dimensions)
//			if(dim>0)
//				++count;
//		projections=new Interface[count];
//		beforeJoining=new Interface[count+1];
//		projectors=new Dense[count];
//		interactions=new PolyInteractionModule[count];
//		offsets=new int[count+1];
//		dims=new int[count];
//		int i=0;
//		int order=1;
//		for(int dim: dimensions) {
//			++order;
//			if(dim<=0)
//				continue;
//			projectors[i]=new Dense();
//			interactions[i]=new PolyInteractionModule(order);
//			dims[i]=dim;
//			++i;
//		}
//
//	}
//
//	@Override
//	public void allocate(InterfaceAllocator ia) {
//		in=ia.get("in");
//		out=ia.get("out");
//		inputWidth=in.count;
//		offsets[0]=inputWidth;
//		for(int i=0; i<interactions.length; ++i) {
//			projections[i]=ia.allocate(new Interface(dims[i], in.tf));
//			projectors[i].allocate(ia);
//		}
//		for(int i=0; i<interactions.length; ++i) {
//			
//			offsets[i+1]=offsets[i]+PolyInteraction.simplexNumber(dims[i], interactions[i].order
//		}
//		
//		exec=new Module[] {
////				trunc, hMat, //Must be first for BP truncation to work
//				xMat, 
//				split,
//				s1, s2, s3, t1,
//				g1, g2, 
//				t2, 
//				g3
//		};
//	}
//
//	@Override
//	public void allocate(ParamAllocator ia) {
//		ia.push(null); xMat.allocate(ia); ia.pop();
//		ia.push(null); hMat.allocate(ia); ia.pop();
//	}
//
//	@Override
//	public void share(ParamBlocks ps) {
//		xMat.share(ps.get("xMat"));
//		hMat.share(ps.get("hMat"));
//		g1.share(ps.get("g1"));
//		g2.share(ps.get("g2"));
//		g3.share(ps.get("g3"));
//		s1.share(ps.get("s1"));
//		s2.share(ps.get("s2"));
//		s3.share(ps.get("s3"));
//		t1.share(ps.get("t1"));
//		t2.share(ps.get("t2"));
//		split.share(ps.get("split"));
//	}
//
//	@Override
//	public ParamBlocks getParamBlocks() {
//		ParamBlocks ret=new ParamBlocks("LSTM");
//		ret.add("xMat", xMat.getParamBlocks());
//		ret.add("hMat", hMat.getParamBlocks());
//		ret.add("g1", g1.getParamBlocks());
//		ret.add("g2", g2.getParamBlocks());
//		ret.add("g3", g3.getParamBlocks());
//		ret.add("s1", s1.getParamBlocks());
//		ret.add("s2", s2.getParamBlocks());
//		ret.add("s3", s3.getParamBlocks());
//		ret.add("t1", t1.getParamBlocks());
//		ret.add("t2", t2.getParamBlocks());
//		ret.add("split", split.getParamBlocks());
//		return ret;
//	}
//
//	@Override
//	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
//		for(Module m: exec)
//			m.forward(training, params, as, t, inst);
//		
//	}
//	@Override
//	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
//		for(int i=exec.length-1; i>=0; --i)
//			exec[i].backprop(phase, params, as, errors, t, inst);
//		
//	}
//	@Override
//	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
//			int t, int[] inst) {
//		for(int i=exec.length-1; i>=0; --i)
//			exec[i].gradients(phase, params, as, errors, gradients, t, inst);
//	}
//	@Override
//	public List<Module> getSubmodules() {
//		return Arrays.asList(exec);
//	}
//
//////	@Override
////	public void declareDependencies(Dependencies d) {
//////		hMat.declareDependencies(d);
//////		xMat.declareDependencies(d);
////		g1.declareDependencies(d);
////		g2.declareDependencies(d);
////		g3.declareDependencies(d);
////		s1.declareDependencies(d);
////		s2.declareDependencies(d);
////		s3.declareDependencies(d);
////		t1.declareDependencies(d);
////		t2.declareDependencies(d);
////		split.declareDependencies(d);
////	}
//
//	@Override
//	public void dontComputeInPhase(String phase) {
//	}
//
////	@Override
//	public boolean wouldBackprop(String phase) {
//		return true;
//	}
//	@Override
//	public int dt() {
//		return 0;
//	}
//	@Override
//	public int[] shift() {
//		return null;
//	}
//
//	@Override
//	public Interface getIn() {
//		return in;
//	}
//
//	@Override
//	public Interface getOut() {
//		return out;
//	}
//	@Override
//	public void initParams(ParamSet p) {
//		defaultInitParams(p);
//		ParamBlock pb=xMat.getBias();
//		int n = pb.count/4;
//		for(int i=0; i<n; ++i) {
//			p.set(pb, i, forgetBiasInit);
//			//			p.set(bnx.add, n+i, 100);
//			//			p.set(bnx.add, 3*n+i, 100);
//		}
//		//		for(int i=0; i<bnh.mult.count; ++i) {
//		//			p.set(bnh.mult, i, p.get(bnh.mult, i)/16);
//		//		}
//	}
//	double forgetBiasInit=1;
//	public Enrichment forgetBiasInit(double d) {
//		forgetBiasInit=d;
//		return this;
//	}
//
//}
