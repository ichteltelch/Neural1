package org.siquod.ml.neural1.modules;

public interface InOutBiasModule extends InOutModule, HasBias{
	@Override
	InOutBiasModule copy();

}
