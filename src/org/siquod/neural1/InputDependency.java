package org.siquod.neural1;

public class InputDependency {
	public final Interface in;
	public final Module via;
	public final int dt;
	public InputDependency(Interface i, Module m, int dt) {
		in=i;
		via=m;
		this.dt=dt;
	}
}
