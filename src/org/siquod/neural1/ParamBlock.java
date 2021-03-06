package org.siquod.neural1;

import java.util.HashSet;

public final class ParamBlock extends Params{
	public int start;
	public int count;
	public ParamBlock(String name, int count){
		super(name);
		this.count=count;
	}
	HashSet<String> dontLearn=new HashSet<>();
	@Override
	public void dontLearnInPhase(String phase) {
		dontLearn.add(phase);
	}
	public boolean shouldLearn(String phase){
		return !dontLearn.contains(phase);
	}
}
