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
import org.siquod.neural1.data.PolyInteraction;
import org.siquod.neural1.data.PolyInteractionFloat;

public class PolyInteractionModule implements InOutModule{
	int order;


	Interface in;
	Interface out;
	public int width=-1;
	public PolyInteractionModule(int order){
		this.order=order;
	}
	public PolyInteractionModule(){
		this(2);
	}
	float[] inBuffer, outBuffer, dInBuffer, dOutBuffer;

	//	public void allocate(InterfaceAllocator ia, String in, String out){
	//		HashMap<String, String> m = new HashMap<>(2);
	//		m.put(in, "in");
	//		m.put(in, "out");
	//		ia.push(m);
	//		allocate(ia);
	//		ia.pop();
	//	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		int incount = in.count;
		int outcount = PolyInteraction.simplexNumber(incount, order);
		inBuffer=new float[incount];
		outBuffer=new float[outcount];
		dInBuffer=new float[incount];
		dOutBuffer=new float[outcount];
		width=incount;
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
		for(ActivationSeq a: as) {
			if(a==null) continue;
			ActivationSet ia=a.get(t);
			if(ia==null)
				continue;
			ActivationSet oa=a.get(t);
			ia.get(in, inBuffer);
			PolyInteractionFloat.apply(width, order, inBuffer, 0, outBuffer, 0);
			oa.add(out, outBuffer);
		}

	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int b=0; b<as.length; ++b) {
			ActivationSeq a=as.a[b];
			if(a==null) continue;
			ActivationSeq e=errors.a[b];
			if(e==null) continue;
			ActivationSet ia=a.get(t);
			if(ia==null) continue;
			ActivationSet ie=e.get(t);
			if(ie==null) continue;
			ActivationSet oe=e.get(t);
			ia.get(in, inBuffer);
			oe.get(out, dOutBuffer);
			PolyInteractionFloat.diffApply(width, order, inBuffer, 0, dInBuffer, 0, dOutBuffer, 0);
			ie.add(in, dInBuffer);
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
