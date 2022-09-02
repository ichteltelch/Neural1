package org.siquod.ml.neural1.modules.loss;

import java.util.HashMap;

import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;

public abstract class LossLayer implements Module{
	
	public void allocate(InterfaceAllocator ia, String in, String target, String loss){
		HashMap<String, String> m = new HashMap<>(3);
		m.put("in", in);
		m.put("target", target);
		m.put("loss", loss);
		ia.push(m);
		allocate(ia);
		ia.pop();
	}
}
