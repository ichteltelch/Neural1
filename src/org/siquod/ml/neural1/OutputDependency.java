package org.siquod.ml.neural1;

public class OutputDependency {
	public final Module via;
	public final Interface out;
	public OutputDependency(Module m, Interface i) {
		out=i;
		via=m;
	}
}
