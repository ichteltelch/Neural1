package org.siquod.ml.data;

import java.util.function.DoubleUnaryOperator;

public class TransformedCursor<BT extends TrainingBatchCursor> implements TrainingBatchCursor {
	protected final BT back;
	protected final CursorTransformer inputTransform;
	protected final CursorTransformer outputTransform;
	protected final DoubleUnaryOperator weightTransform;
	protected final int inputCount;
	protected final int outputCount;
	
	public TransformedCursor(BT back, CursorTransformer inputTransformer, CursorTransformer outputTransform, DoubleUnaryOperator weightTransformer) { 
		this.back = back;
		this.inputTransform=inputTransformer;
		this.outputTransform=outputTransform;
		this.weightTransform=weightTransformer;
		this.inputCount = inputTransform==null? back.inputCount() : Math.max(back.inputCount(), inputTransform.dimAfterTransform(back.inputCount()));
		this.outputCount = outputTransform==null? back.outputCount() : Math.max(back.outputCount(), outputTransform.dimAfterTransform(back.outputCount()));
	}

	@Override public int outputCount() {return outputCount;}

	@Override public int inputCount() {return inputCount;}

	@Override public void giveOutputs(double[] outputs) {
		back.giveOutputs(outputs);
		if(outputTransform!=null)
			outputTransform.transform(outputs);
	}

	@Override public void giveInputs(double[] inputs) {
		back.giveInputs(inputs);
		if(inputTransform!=null)
			inputTransform.transform(inputs);
	}

	@Override public double getWeight() {
		double w = back.getWeight();
		return weightTransform==null?w:weightTransform.applyAsDouble(w);
	}

	@Override public void reset() {back.reset();}

	@Override public void next() {back.next();}

	@Override public boolean isFinished() {return back.isFinished();}
	@SuppressWarnings("unchecked")
	@Override public TransformedCursor<BT> clone() {
		BT bc;
		bc = (BT) back.clone();
		return new TransformedCursor<BT>(bc, inputTransform, outputTransform, weightTransform);
	}
}