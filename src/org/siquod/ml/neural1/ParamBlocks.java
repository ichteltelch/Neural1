package org.siquod.ml.neural1;

import java.util.HashMap;

public class ParamBlocks extends Params{
	HashMap<String, ParamBlock> sub;
	HashMap<String, ParamBlocks> subs;
	public ParamBlocks(String name) {
		super(name);
		sub=new HashMap<>();
		subs=new HashMap<>();
	}
	public void add(String name, ParamBlock p){
		sub.put(name, p);
	}
	public void add(String name, ParamBlocks p){
		subs.put(name, p);
	}
	public void add(ParamBlock p){
		sub.put(p.name, p);
	}
	public void add(ParamBlocks p){
		subs.put(p.name, p);
	}
	public ParamBlock get(String name, int size) {
		ParamBlock ret = sub.get(name);
		if(ret==null)
			throw new IllegalArgumentException("The parameters of '"+this.name+"' do not contain a block called '"+name+"'");
		if(ret.count!=size)
			throw new IllegalArgumentException("The param block '"+name+"' of '"+this.name+" does not have the expected size");
		return ret;
	}
	public ParamBlocks get(String name) {
		ParamBlocks ret = subs.get(name);
		if(ret==null)
			if(!subs.containsKey(name))
				throw new IllegalArgumentException("The parameters of '"+this.name+"' do not contain a subcompound called '"+name+"'");
		return ret;
	}
	@Override
	public void dontLearnInPhase(String phase) {
		for(ParamBlock p: sub.values())
			p.dontLearnInPhase(phase);
		for(ParamBlocks p: subs.values())
			p.dontLearnInPhase(phase);
	}
}
