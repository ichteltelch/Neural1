package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamBlock;

public interface HasScale extends Module {
	public ParamBlock getScale();
}
