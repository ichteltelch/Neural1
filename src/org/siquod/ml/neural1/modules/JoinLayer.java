package org.siquod.ml.neural1.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.NopCastLayer;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
/**
 * A JoinLayer doesn't do anything at runtime, but it declares input 
 * interfaces that provide aliasing views on portions of its single output interface.
 * Note that that the output layer must be allocated before allocate() is called, 
 * and the input layers will be allocated only afterwards.
 * @author bb
 *
 */
public class JoinLayer implements NopCastLayer{
	Interface out;
	ArrayList<Interface> in=new ArrayList<>();
	ArrayList<Integer> offsets=new ArrayList<>();
	public JoinLayer(){
		
	}
	public JoinLayer input(int offset, Interface i){
		if(i==null)
			throw new NullPointerException();
		if(offset<0)
			throw new IllegalArgumentException("offset must be positive");
		offsets.add(offset);
		in.add(i);
		return this;
	}
	public Interface getInput(int i){
		return in.get(i);
	}
	public void allocate(InterfaceAllocator ia, String out){
		HashMap<String, String> m = new HashMap<>(1);
		m.put("out", out);
		ia.push(m);
		allocate(ia);
		ia.pop();
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		out=ia.get("out");
		for(int i=0; i<in.size(); ++i){
			Interface o=in.get(i);
			o.offset = out.offset + offsets.get(i);
			if(o.offset+o.count>out.count)
				throw new IllegalArgumentException("output layer is to small");
		}
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

	}
	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		
	}
////	@Override
//	public void declareDependencies(Dependencies d) {
//		for(Interface i: in)
//			d.declare(new InputDependency(i, this, 0));
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
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}
}
