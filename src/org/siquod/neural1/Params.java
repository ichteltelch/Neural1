package org.siquod.neural1;

public abstract class Params {
	public final String name;
	public Params(String name) {
		this.name=name;
	}
	public abstract void dontLearnInPhase(String phase);

}
