package org.siquod.neural1;

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
	
}
