package org.siquod.ml.metrics;

class MA_MetricTracker extends MetricTracker{

	int past;
	int future;
	int[] iterationBuffer;
	double[] valueBuffer;
	int count;
	int offset;
	double sum;
	int refreshCounter;
	
	
	public MA_MetricTracker(String id, Metric m, int past, int future, DerivableMetricTracker source) {
		super(id, m.movingAverage(past, future), source);
		int l = past+future+1;
		iterationBuffer = new int[l];
		valueBuffer = new double[past+future+1];
		this.past = past;
		this.future = future;
	}

	@Override
	public void observe(DerivableMetricTracker source, int iteration, double value) {
		if(count>=valueBuffer.length) {
			assert count==valueBuffer.length;
			offset = offset%valueBuffer.length;				
			sum -= valueBuffer[offset];
			--count;
			++offset;
		}
		int newIndex = (offset + count)%valueBuffer.length;
		++count;
		iterationBuffer[newIndex] = iteration;
		valueBuffer[newIndex] = value;
		sum += value;
		if(++refreshCounter == 1024) {
			refreshCounter = 0;
			sum=0;
			for(int i=0; i<count; ++i)
				sum += valueBuffer[(offset + i)%valueBuffer.length];
		}
		if(count==valueBuffer.length) {
			observe(iterationBuffer[(offset + past)%iterationBuffer.length], sum/count);
		}
		
	}

	@Override
	protected void doObserve(int iteration, double value) {
		
	}
	
}