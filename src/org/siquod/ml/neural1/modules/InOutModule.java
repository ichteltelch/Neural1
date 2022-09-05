package org.siquod.ml.neural1.modules;

import java.util.HashMap;

import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;

public interface InOutModule extends Module{
	public default void allocate(InterfaceAllocator ia, String in, String out){
		HashMap<String, String> m = new HashMap<>(2);
		m.put("in", in);
		m.put("out", out);
		ia.push(m);
		allocate(ia);
		ia.pop();
	}
	public default InOutModule shift(int... shift){
		throw new UnsupportedOperationException("Cannot modify shift of this module");
	}
	public default InOutModule dt(int dt){
		throw new UnsupportedOperationException("Cannot modify time shift of this module");
	}
	public abstract Interface getIn();
	public abstract Interface getOut();
	public abstract int dt() ;
	public abstract int[] shift() ;

}
