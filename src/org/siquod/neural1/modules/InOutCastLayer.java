package org.siquod.neural1.modules;

import org.siquod.neural1.InputDependency;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.OutputDependency;


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
