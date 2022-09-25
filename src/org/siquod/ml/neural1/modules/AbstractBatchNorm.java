package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ParamBlock;
import org.siquod.ml.neural1.ParamSet;

public abstract class AbstractBatchNorm implements InOutScaleBiasModule {
	public abstract ParamBlock sigmaSotrage();
	public abstract ParamBlock meanStorage();
	public abstract double mapSigmaFromStorage(double s);
	public abstract double mapSigmaToStorage(double s);
	public abstract double distributionMismatch(ActivationBatch as, ParamSet params, int t);
}
