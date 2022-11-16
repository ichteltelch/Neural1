package org.siquod.ml.metrics;

import java.util.Arrays;

class MRA_MetricTracker extends MetricTracker{

	int past;
	int future;
	int[] iterationBuffer;
	double[] valueBuffer;
	double[] tmpBuffer;
	int count;
	int offset;
	int refreshCounter;
	int minIndex;
	int maxIndex;
	
	
	public MRA_MetricTracker(String id, Metric m, int past, int future, double fraction, DerivableMetricTracker source) {
		super(id, m.movingRobustAverage(past, future, fraction), source);
		if(!(fraction>0 && fraction<=1))
			throw new IllegalArgumentException();
		int l = past+future+1;
		iterationBuffer = new int[l];
		valueBuffer = new double[past+future+1];
		tmpBuffer = new double[past+future+1];
		this.past = past;
		this.future = future;
		minIndex = Math.max(0, (int)(Math.floor((past + future)*0.5*(1-fraction))));
		maxIndex = Math.min(past+future, (int)(Math.ceil ((past + future)*0.5*(1+fraction))));
	}

	@Override
	public void observe(DerivableMetricTracker source, int iteration, double value) {
		if(count>=valueBuffer.length) {
			assert count==valueBuffer.length;
			offset = offset%valueBuffer.length;				
			--count;
			++offset;
		}
		int newIndex = (offset + count)%valueBuffer.length;
		++count;
		iterationBuffer[newIndex] = iteration;
		valueBuffer[newIndex] = value;

		if(count==valueBuffer.length) {
			System.arraycopy(valueBuffer, 0, tmpBuffer, 0, valueBuffer.length);
			Arrays.sort(tmpBuffer);
			double sum = 0;
			for(int i=minIndex; i<=maxIndex; ++i) {
				sum += tmpBuffer[i];
			}
			observe(iterationBuffer[(offset + past)%iterationBuffer.length], sum / (maxIndex + 1 - minIndex) );
		}
		
	}

	@Override
	protected void doObserve(int iteration, double value) {
		
	}
	
}