package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;


public abstract class InOutCastLayer implements InOutModule{
	Interface in;
	Interface out;
	public InOutCastLayer(Interface in2) {
		in=in2;
	}
	public Interface getOutput(){
		return out;
	}
	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		if(in.offset!= out.offset)
			throw new IllegalStateException("This is a cast layer");
	}
////	@Override
//	public void declareDependencies(Dependencies d) {
//		d.declare(new InputDependency(in, this, 0));
//		d.declare(new OutputDependency(this, out));
//	}
}
