package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.ParamBlock;
import org.siquod.ml.neural1.ParamSet;

public abstract class BatchNormoid extends AbstractBatchNorm{
	public static enum FinalizationMode{
		NORMAL,
		NONE,
		MEANS,
		STANDARD_DEVIATIONS
	}
	protected ParamBlock runningMean, runningSdev;

	protected FinalizationMode finalizationMode = FinalizationMode.NORMAL;
	int instanceCount;
	public void setMode(FinalizationMode m, ParamSet p) {
		if(finalizationMode==m)
			return;
		switch(finalizationMode) {
		case NONE: break;
		case NORMAL: break;
		case MEANS: 
			for(int i=0; i<runningMean.count; ++i) {
				p.set(runningMean, i, p.get(runningMean, i)/instanceCount);
			}
			break;
		case STANDARD_DEVIATIONS:
			for(int i=0; i<runningSdev.count; ++i) {
				p.set(runningSdev, i, Math.sqrt(p.get(runningSdev, i)/(instanceCount-1)));
			}
			break;
		}
		
		switch(m) {
		case NONE: break;
		case NORMAL: break;
		case MEANS: 
			instanceCount=0;
			for(int i=0; i<runningMean.count; ++i) {
				p.set(runningMean, i, 0);
			}
			break;
		case STANDARD_DEVIATIONS:
			instanceCount=0;
			for(int i=0; i<runningSdev.count; ++i) {
				p.set(runningSdev, i, 0);
			}
			break;
		}
	
		finalizationMode = m;

	}
	@Override
	public double mapSigmaFromStorage(double s) {
		return s;
	}
	@Override
	public double mapSigmaToStorage(double s) {
		return s;
	}
	@Override
	public ParamBlock sigmaStorage() {
		return runningSdev;
	}
	@Override
	public ParamBlock meanStorage() {
		return runningMean;
	}
}
