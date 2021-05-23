package org.siquod.neural1.modules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.InputDependency;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.NopCastLayer;
import org.siquod.neural1.OutputDependency;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;

/**
 * A SplitLayer doesn't do anything at runtime, but it declares output 
 * interfaces that provide aliasing views on portions of its single input interface
 * @author bb
 *
 */
public class SplitLayer implements NopCastLayer{
	Interface in;
	ArrayList<Interface> out=new ArrayList<>();
	ArrayList<Integer> offsets=new ArrayList<>();
	public SplitLayer(){

	}
	public SplitLayer output(int offset, Interface i){
		if(i==null)
			throw new NullPointerException();
		if(offset<0)
			throw new IllegalArgumentException("offset must be positive");
		offsets.add(offset);
		out.add(i);
		return this;
	}
	public Interface getOutput(int i){
		return out.get(i);
	}
	public void allocate(InterfaceAllocator ia, String in){
		HashMap<String, String> m = new HashMap<>(1);
		m.put("in", in);
		ia.push(m);
		allocate(ia);
		ia.pop();
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		for(int i=0; i<out.size(); ++i){
			Interface o=out.get(i);
			o.offset = in.offset + offsets.get(i);
			if(o.offset+o.count>in.offset+in.count)
				throw new IllegalArgumentException("input layer is to small");
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
//		d.declare(new InputDependency(in, this, 0));
//		for(Interface o: out)
//			d.declare(new OutputDependency(this, o));			
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
