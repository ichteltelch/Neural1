package org.siquod.ml.data;

/**
 * An interface for accessing training data samples for machine learning and model fitting.
 * The data presented by this object may change if an appropriate 
 * method of the implementing class is called, 
 * such as {@link TrainingBatchCursor#next()} or {@link TrainingBatchCursor#reset()}
 * @author bb
 *
 */
public interface TrainingDataGiver {
	/**
	 * The number of input variables of a training example
	 * @return
	 */
	public int inputCount();
	/**
	 * The number of output variables of a training example
	 * @return
	 */
	public int outputCount();
	/**
	 * Write the input variable values into the given array
	 * @param inputs
	 */
	public void giveInputs(double[] inputs);
	/**
	 * Write the output variable values into the given array
	 * @param inputs
	 */
	public void giveOutputs(double[] outputs);
	/**
	 * 
	 * @return The weight of the current training example
	 */
	public double getWeight();
	
	public default TrainingDataGiver whitened(Whitener whitenInputs, Whitener whitenOutputs) {
		return new WhitenedTrainingDataGiver<TrainingDataGiver>(this, whitenInputs, whitenOutputs);
	}
	static public class WhitenedTrainingDataGiver<B extends TrainingDataGiver> implements TrainingDataGiver{
		final Whitener whitenInputs;
		final Whitener whitenOutputs;
		final double[] inputBuffer;
		final double[] outputBuffer;
		final B back;
		public WhitenedTrainingDataGiver(B back, Whitener whitenInputs, Whitener whitenOutputs) {
			this.back=back;
			this.whitenInputs=whitenInputs;
			this.whitenOutputs=whitenOutputs;
			if(whitenInputs!=null) {
				if(back.inputCount()!=whitenInputs.dim())
					throw new IllegalArgumentException("dimension mismatch");
				inputBuffer=new double[back.inputCount()];
			}else
				inputBuffer=null;
			if(whitenOutputs!=null) {
				if(back.outputCount()!=whitenOutputs.dim())
					throw new IllegalArgumentException("dimension mismatch");
				outputBuffer=new double[back.outputCount()];
			}else
				outputBuffer=null;
		}
		
		@Override
		public int inputCount() {
			return back.inputCount();
		}

		@Override
		public int outputCount() {
			return back.outputCount();
		}

		@Override
		public void giveInputs(double[] inputs) {
			if(whitenInputs==null)
				back.giveInputs(inputs);
			else {
				back.giveInputs(inputBuffer);
				whitenInputs.whiten(inputBuffer, inputs);
			}	
		}

		@Override
		public void giveOutputs(double[] outputs) {
			if(whitenOutputs==null)
				back.giveOutputs(outputs);
			else {
				back.giveOutputs(outputBuffer);
				whitenOutputs.whiten(outputBuffer, outputs);
			}		
		}

		@Override
		public double getWeight() {
			return back.getWeight();
		}
		
	}
	
	
	
}
