package org.siquod.neural1;

/**
 * A {@link TrainingBatchCursor} presents a sequence of training data items to a
 * machine learning or model fitting algorithm 
 * through the methods inherited from {@link TrainingDataGiver}.
 * If the end of the sequence has been reached (as told by {@link #isFinished()},
 * these methods should not be called anymore until {@link #reset()} has been called
 * @author bb
 *
 */
public interface TrainingBatchCursor extends TrainingDataGiver{
	public static final class SingletonBatchCursorSingleOutput implements TrainingBatchCursor {
		private final double output;
		private final double[] inputs;
		public double weight;
		boolean consumed=false;

		public SingletonBatchCursorSingleOutput(double output, double[] inputs, double weight) {
			this.output = output;
			this.inputs = inputs;
			this.weight = weight;
		}

		@Override public int outputCount() {return 1;}

		@Override public int inputCount() {return inputs.length;}

		@Override public void giveOutputs(double[] outputs0) {
			outputs0[0]=output;
		}

		@Override public void giveInputs(double[] inputs0) {
			System.arraycopy(inputs, 0, inputs0, 0, inputs.length);
		}

		@Override public double getWeight() {return weight;}

		@Override
		public void reset() {
			consumed=false;
		}

		@Override
		public void next() {
			consumed=true;
		}

		@Override
		public boolean isFinished() {
			return consumed;
		}
	}
	public static final class SingletonBatchCursor implements TrainingBatchCursor {
		private final double[] inputs;
		private final double[] outputs;
		public double weight;
		boolean consumed=false;

		public SingletonBatchCursor(double[] inputs, double[] outputs, double weight) {
			this.inputs = inputs;
			this.outputs = outputs;
			this.weight = weight;
		}

		@Override public int outputCount() {return outputs.length;}

		@Override public int inputCount() {return inputs.length;}

		@Override public void giveOutputs(double[] outputs0) {
			System.arraycopy(outputs, 0, outputs0, 0, outputs.length);
		}

		@Override public void giveInputs(double[] inputs0) {
			System.arraycopy(inputs, 0, inputs0, 0, inputs.length);
		}

		@Override public double getWeight() {return weight;}

		@Override
		public void reset() {
			consumed=false;
		}

		@Override
		public void next() {
			consumed=true;
		}

		@Override
		public boolean isFinished() {
			return consumed;
		}
	}
	/**
	 * Go to the next data item, or reach the end of the sequence
	 */
	public void next();
	/**
	 * @return whether the end of the sequence has been reached
	 */
	public boolean isFinished();
	/**
	 * Restart the sequence. After calling this method, this
	 * {@link TrainingBatchCursor} must present the same items as it did the first time through.
	 * (Although subclasses may choose to reload the sequence with different data 
	 * between calls to the machine learning algorithm. But while the algorithm is processing
	 * the batch, the sequence should not change)
	 */
	public void reset();
	
	/**
	 * Concatenate several sequences. 
	 * The sequences must have the same format (# of input and output variables).
	 * Even though I can't imagine a use case, you can concatenate a sequence with itself
	 * @param sequences
	 * @return
	 */
	public static TrainingBatchCursor concat(TrainingBatchCursor... sequences) {
		if(sequences.length==0) {
			throw new IllegalArgumentException("Must concatenate a nonzero number of sequences");
		}
		if(sequences.length==1)
			return sequences[1];
		int inDim = sequences[0].inputCount();
		int outDim = sequences[0].outputCount();
		for(int i=1; i<sequences.length; ++i) {
			TrainingBatchCursor it = sequences[i];
			if(it.inputCount()!=inDim)
				throw new IllegalArgumentException("Sequence #"+i+" has a different number of input variables than sequence #0");
			if(it.outputCount()!=outDim)
				throw new IllegalArgumentException("Sequence #"+i+" has a different number of output variables than sequence #0");
		}
		TrainingBatchCursor ret = new TrainingBatchCursor() {
			int currentIndex = 0;
			@Override public int outputCount() {return outDim;}
			@Override public int inputCount() {return inDim;}			
			@Override
			public void giveOutputs(double[] outputs) {
				if(isFinished())
					throw new IllegalStateException("This cursor has reached its end.");
				sequences[currentIndex].giveOutputs(outputs);
			}
			
			@Override
			public void giveInputs(double[] inputs) {
				if(isFinished())
					throw new IllegalStateException("This cursor has reached its end.");
				sequences[currentIndex].giveInputs(inputs);
			}
			@Override
			public double getWeight() {
				if(isFinished())
					throw new IllegalStateException("This cursor has reached its end.");
				return sequences[currentIndex].getWeight();
			}
			
			@Override
			public void reset() {
				sequences[0].reset();
				currentIndex=0;
				ff();
			}
			private void ff() {
				while(sequences[currentIndex].isFinished()) {
					++currentIndex;
					if(currentIndex>=sequences.length)
						break;
					sequences[currentIndex].reset();
				}
			}
			
			@Override
			public void next() {
				if(isFinished())				
					throw new IllegalStateException("This cursor has reached its end.");
				sequences[currentIndex].next();
				ff();
			}
			
			@Override
			public boolean isFinished() {
				return currentIndex>=sequences.length;
			}
	
		};
		ret.reset();
		return ret;
	}
	/**
	 * Make a sequence containing one element
	 * @param inputs This array is not copied, so if you change it, you change the singleton
	 * @param outputs This array is not copied, so if you change it, you change the singleton
	 * @param weight
	 * @return
	 */
	public static TrainingBatchCursor singleton(double[] inputs, double[] outputs, double weight) {
		return new SingletonBatchCursor(inputs, outputs, weight);
	}
	/**
	 * Make a sequence containing one element that has a single output
	 * @param inputs This array is not copied, so if you change it, you change the singleton
	 * @param output This array is not copied, so if you change it, you change the singleton
	 * @param weight
	 * @return
	 */
	public static TrainingBatchCursor singleton(double[] inputs, double output, double weight) {
		return new SingletonBatchCursorSingleOutput(output, inputs, weight);
	}
	/**
	 * Make a wrapper for a {@link TrainingBatchCursor} that enriches its inputs by
	 * taking polynomial combinations of the original data's inputs up to a given order
	 * 
	 * @param back
	 * @param order
	 * @return
	 */
	public static TrainingBatchCursor polyInteractionFeatures(TrainingBatchCursor back, int order) {
		int bic = back.inputCount();
		int eInp = bic;
		for(int i=2; i<order; ++i)
			eInp+=PolyInteraction.simplexNumber(bic, i);
		int inputCount = eInp;
		return new TrainingBatchCursor() {
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
		};
		
		
	}
}
