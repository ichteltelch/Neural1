package org.siquod.neural1.modules;

import org.siquod.neural1.Module;
import org.siquod.neural1.ParamBlock;

public interface HasBias extends Module {
	public ParamBlock getBias();
}
