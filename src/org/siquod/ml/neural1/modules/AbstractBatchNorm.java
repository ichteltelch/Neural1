package org.siquod.ml.neural1.modules;

import org.siquod.ml.data.TrainingBatchCursor;
import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ActivationSeq;
import org.siquod.ml.neural1.ActivationSet;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.ParamBlock;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;
import org.siquod.ml.neural1.net.FeedForward.Eval;

public abstract class AbstractBatchNorm implements InOutScaleBiasModule {
	public static double defaultGradMult = 1;
	public static double defaultLrMult = .1;
	public abstract ParamBlock sigmaStorage();
	public abstract ParamBlock meanStorage();
	public abstract double mapSigmaFromStorage(double s);
	public abstract double mapSigmaToStorage(double s);
	public abstract double distributionMismatch(ActivationBatch as, ParamSet params, int t);
	public abstract float learnRateMultiplier();
	@Override
	public void setLearnRateMultipliers(ParamSet lrm) {
		InOutScaleBiasModule.super.setLearnRateMultipliers(lrm);
		float m = learnRateMultiplier();
		lrm.setAll(sigmaStorage(), m);
		lrm.setAll(meanStorage(), m);
	}
	abstract TensorFormat tf();
	public class StatRecorder{
		double[] sum = new double[tf().channels()];
		double[] squares = new double[tf().channels()];
		int count;
		public void record(ActivationBatch as, int t) {
			TensorFormat tf = tf();
			int maxBri = tf.dims[0];
			Interface in = getIn();
			for(ActivationSeq s: as) {
				ActivationSet a = s.get(t);
				for(int bri=0; bri<maxBri; ++bri) {
					for(int chan=0; chan<sum.length; ++chan) {
						float v = a.get(in, tf.index(bri, chan));
						sum[chan] += v;
						squares[chan] += v*v;
					}
					++count;
				}
			}
		}
		public void setNormalizerParameters(ParamSet params) {
			ParamBlock ss = sigmaStorage();
			ParamBlock ms = meanStorage();
			for(int i=0; i<sum.length; ++i) {
				double mean = sum[i]/count;
				params.set(ms, i, mean);
				double var = Math.abs(squares[i]/count - mean*mean);
				params.set(ss, i, mapSigmaToStorage(Math.sqrt(var)));
				
			}
			
		}
	}
	public static void setNormalizerParameters(Eval eval, int bs, TrainingBatchCursor data, Iterable<? extends AbstractBatchNorm> bns) {
		for(AbstractBatchNorm bn: bns) {
			StatRecorder rec = bn.new StatRecorder();
			eval.computeLoss(data, bs, false, ass->{
				rec.record(ass, 0);
			});
			rec.setNormalizerParameters(eval.getParams());
		}
	}
	@Override
	public abstract AbstractBatchNorm copy();
}
