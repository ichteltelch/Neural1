package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamBlock;

public interface HasBias extends Module {
	public ParamBlock getBias();
}
