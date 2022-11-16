package org.siquod.ml.metrics;

import java.util.Arrays;

class MM_MetricTracker extends MetricTracker{

	int past;
	int future;
	int[] iterationBuffer;
	double[] valueBuffer;
	double[] tmpBuffer;
	int count;
	int offset;
	int refreshCounter;
	
	
	public MM_MetricTracker(String id, Metric m, int past, int future, DerivableMetricTracker source) {
		super(id, m.movingMedian(past, future), source);
		int l = past+future+1;
		iterationBuffer = new int[l];
		valueBuffer = new double[past+future+1];
		tmpBuffer = new double[past+future+1];
		this.past = past;
		this.future = future;
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
			int medi = tmpBuffer.length>>1;
			double median = (tmpBuffer.length&1)==0?(tmpBuffer[medi]+tmpBuffer[medi-1])*0.5:tmpBuffer[medi];
			observe(iterationBuffer[(offset + past)%iterationBuffer.length], median);
		}
		
	}

	@Override
	protected void doObserve(int iteration, double value) {
		
	}
	
}