package org.siquod.ml.neural1.modules;

public interface InOutScaleBiasModule extends InOutBiasModule, HasScale{
	@Override InOutScaleBiasModule copy();

}
