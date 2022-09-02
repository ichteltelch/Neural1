package org.siquod.ml.data;


public class PolyInteractionCursor<BT extends TrainingBatchCursor> implements TrainingBatchCursor {
	protected final int inputCount;
	protected final int order;
	protected final BT back;
	protected final int bic;

	public PolyInteractionCursor(int inputCount, int order, BT back, int bic) {
		this.inputCount = inputCount;
		this.order = order;
		this.back = back;
		this.bic = bic;
	}

	@Override public int outputCount() {return back.outputCount();}

	@Override public int inputCount() {return inputCount;}

	@Override public void giveOutputs(double[] outputs) {back.giveOutputs(outputs);}

	@Override public void giveInputs(double[] inputs) {
		back.giveInputs(inputs);
		PolyInteraction.apply(bic, 2, order, inputs, 0, inputs, bic);
	}

	@Override public double getWeight() {return back.getWeight();}

	@Override public void reset() {back.reset();}

	@Override public void next() {back.next();}

	@Override public boolean isFinished() {return back.isFinished();}
	@SuppressWarnings("unchecked")
	@Override public PolyInteractionCursor<BT> clone() {
		BT bc;
		bc = (BT) back.clone();
		return new PolyInteractionCursor<BT>(inputCount, order, bc, bic);
	}
}