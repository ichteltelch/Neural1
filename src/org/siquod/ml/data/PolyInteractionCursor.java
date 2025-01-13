package org.siquod.ml.data;


public class PolyInteractionCursor<BT extends TrainingBatchCursor> implements TrainingBatchCursor {
	protected final int inputCount;
	protected final int order;
	protected final BT back;
	protected final int interactingFeatures;
	protected final int interactedFeatures;
	protected final int prefixLength;
	protected final int suffixLength;
	protected final int inSuffixOffset;
	protected final int outSuffixOffset;
	protected final int outputCount;
	
	public PolyInteractionCursor(BT back, int order, int prefixLength, int suffixLength) { 
		this.back = back;
		this.prefixLength = prefixLength;
		this.suffixLength = suffixLength;
		this.interactingFeatures = back.inputCount() - (prefixLength + suffixLength);
		this.interactedFeatures = PolyInteraction.simplexNumberSum(interactingFeatures, 1, order);		
		this.inputCount = interactedFeatures + prefixLength + suffixLength;
		this.inSuffixOffset = prefixLength + interactingFeatures;
		this.outSuffixOffset = prefixLength + interactedFeatures;
		this.order = order;
		this.outputCount = back.outputCount();
	}

	@Override public int outputCount() {return back.outputCount();}

	@Override public int inputCount() {return inputCount;}

	@Override public void giveOutputs(double[] outputs) {back.giveOutputs(outputs);}

	@Override public void giveInputs(double[] inputs) {
		back.giveInputs(inputs);
		if(suffixLength!=0)
			System.arraycopy(inputs, inSuffixOffset, inputs, outSuffixOffset, suffixLength);
		PolyInteraction.apply(interactedFeatures, 2, order, inputs, prefixLength, inputs, prefixLength + interactingFeatures);
	}

	@Override public double getWeight() {return back.getWeight();}

	@Override public void reset() {back.reset();}

	@Override public void next() {back.next();}

	@Override public boolean isFinished() {return back.isFinished();}
	@SuppressWarnings("unchecked")
	@Override public PolyInteractionCursor<BT> clone() {
		BT bc;
		bc = (BT) back.clone();
		return new PolyInteractionCursor<BT>(bc, order, prefixLength, suffixLength);
	}
}