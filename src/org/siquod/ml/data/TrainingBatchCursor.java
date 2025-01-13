package org.siquod.ml.data;






import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * A {@link TrainingBatchCursor} presents a sequence of training data items to a
 * machine learning or model fitting algorithm 
 * through the methods inherited from {@link TrainingDataGiver}.
 * If the end of the sequence has been reached (as told by {@link #isFinished()},
 * these methods should not be called anymore until {@link #reset()} has been called
 * @author bb
 *
 */
public interface TrainingBatchCursor extends TrainingDataGiver, Cloneable{
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
	 * Make a copy of this TrainingBatchCursor that can be iterated over independently
	 * @return
	 */
	TrainingBatchCursor clone();
	/**
	 * A {@link TrainingBatchCursor} that contains a fixed known number of items
	 * and that can be positioned quickly at each item
	 * @author bb
	 *
	 */
	public interface RandomAccess extends TrainingBatchCursor{
		/**
		 * How many items are in this cursor
		 * @return
		 */
		long size();
		/**
		 * go to a certain position
		 * @param position
		 */
		void seek(long position);
		/**
		 * Make a copy of this cursor that can be iterated independently.
		 * The copy may or may not be in the reset state initially.
		 */
		RandomAccess clone();
		/**
		 * Split this cursor into several cursors that iterate over 
		 * subsequences of approximately equal length
		 * @param parts The number of subsequences to generate
		 * @see #subsequence(long, long)
		 * @see #split(double...)
		 * @return
		 */
		default RandomAccess[] split(int parts) {
			RandomAccess[] ret = new RandomAccess[parts];
			long total = size();
			for(int part=0; part<parts; ++part)
				ret[part]=subsequence(((total*part)/parts), ((total*(part+1))/parts));
			return ret;
		}
		/**
		 * Split this cursor into several cursors that iterate over 
		 * subsequences whose lengths are proportional to the given numbers
		 * @param ratios The proportional length of each subsequence to generate
		 * @see #subsequence(long, long)
		 * @see #split(int)
		 * @return
		 */
		default RandomAccess[] split(double... ratios){
			RandomAccess[] ret = new RandomAccess[ratios.length];
			double ratioSum=0;
			for(int i=0; i<ratios.length; ++i)
				ratioSum+=ratios[i];
			long size = size();
			double renorm = size/ratioSum;

			long[] endIndices=new long[ratios.length];
			{
				double lastEndIndex=0;
				for(int i=0; i<ratios.length; ++i) {
					double nextEndIndex=lastEndIndex + renorm*ratios[i];
					endIndices[i]=(int)Math.round(nextEndIndex);
					lastEndIndex=nextEndIndex;
				}
				endIndices[ratios.length-1]=size;
			}

			long lastEndIndex=0;
			for(int i=0; i<ratios.length; ++i) {
				long nextEndIndex=endIndices[i];
				RandomAccess relem = subsequence(lastEndIndex, nextEndIndex);
				ret[i]=relem;
				lastEndIndex=nextEndIndex;
			}
			return ret;
		}
		
		/**
		 * Create an independent cursor over the same data that is restricted to a 
		 * subrange of the item indices of this one
		 * @param start Starting position. If negative, 0 will be assumed
		 * @param end End position (exclusive). If greater than the number of elements, 
		 * the number of elements will be assumed. If less then {@code start}, an empty cursor 
		 * will be returned
		 * @return
		 */
		public default RandomAccess subsequence(long start, long end) {
			if(start<0)
				start=0;
			if(end>size())
				end=size();
			if(end<=start)
				return empty(inputCount(), outputCount());
			long fEnd=end;
			long fStart=start;
			RandomAccess orig=clone();
			int ic = inputCount();
			int oc = outputCount();
			return new RandomAccess() {
				long at=0;
				@Override
				public RandomAccess clone() {
					return orig.clone().subsequence(fStart, fEnd);
				}
				@Override public double getWeight() {return orig.getWeight();}
				@Override public void giveInputs(double[] inputs) {orig.giveInputs(inputs);}
				@Override public void giveOutputs(double[] outputs) {orig.giveOutputs(outputs);}
				@Override public int inputCount() {return ic;}
				@Override public int outputCount() {return oc;}
				@Override public boolean isFinished() {return at>=size();}
				@Override public void next() {++at; orig.next();}
				@Override public void reset() {at=0; orig.seek(fStart);}
				@Override public void seek(long position) {
					if(position<0)
						throw new IllegalArgumentException("Seek position must mut be negative");
					at=position; 
					orig.seek(fStart+position);
				}
				@Override public long size() {return fEnd-fStart;}
				@Override
				public RandomAccess subsequence(long start, long end) {
					if(start<0)
						start=0;
					if(end>size())
						end=size();
					if(end<=start)
						return empty(ic, oc);
					return orig.subsequence(fStart+start, fStart+end);
				}
			};
		}
		@Override
		default RandomAccess whitened(Whitener whitenInputs, Whitener whitenOutputs) {
			return new WhitenedRandomAccess<RandomAccess>(this, whitenInputs, whitenOutputs);
		}
		static public class WhitenedRandomAccess<B extends RandomAccess> 
		extends WhitenedTrainingBatchCursor<B>
		implements RandomAccess{

			public WhitenedRandomAccess(B back, Whitener whitenInputs, Whitener whitenOutputs) {
				super(back, whitenInputs, whitenOutputs);
			}

			

			@SuppressWarnings("unchecked")
			@Override
			public WhitenedRandomAccess<B> clone() {
				return new WhitenedRandomAccess<B>((B)((RandomAccess)back).clone(), whitenInputs, whitenOutputs);
			}



			@Override
			public long size() {
				return back.size();
			}



			@Override
			public void seek(long position) {
				back.seek(position);
			}
			
		}
		public default RandomAccess fillToModulus(int modulus, boolean roundUpNotDown) {
			long size = this.size();
			
			int repeats;
			if(size>=modulus) {
				repeats = (size%modulus==0)?1:2;
			}else {
				long count = size/modulus;

				if(!roundUpNotDown || size%modulus==0)
					repeats = (int)count;
				else
					repeats = (int) (count + 1);
			}
			RandomAccess repeated;
			if(repeats>1) {
				RandomAccess[] rs = new RandomAccess[repeats];
				Arrays.fill(rs, this);
				repeated = concat(rs);
			}else {
				repeated = this;
			}
			RandomAccess truncated;
			long over = repeated.size() % modulus;
			if(over==0) {
				truncated = repeated;
			}else {
				truncated = repeated.subsequence(0, repeated.size() - over);
			}
			return truncated;
		}
		default RandomAccess remap(double weightScale, int[] outputMapping, int... inputMapping) {
			return new RemappedCursor.RandomAccess(this, weightScale, outputMapping, inputMapping);
		}
		default RandomAccess scaleWeights(double weightScale) {
			return new RemappedCursor.RandomAccess(this, weightScale, null, null);
		}
		
		public default RandomAccess transformInputs(CursorTransformer inT) {
			return new TransformedCursor_RA(this, inT, null, null);
		}
		public default RandomAccess transformInputs(CursorTransformer inT, DoubleUnaryOperator weightT) {
			return new TransformedCursor_RA(this, inT, null, weightT);
		}
		public default RandomAccess transformOutputs(CursorTransformer outT) {
			return new TransformedCursor_RA(this, null, outT, null);
		}
		public default RandomAccess transformOutputs(CursorTransformer outT, DoubleUnaryOperator weightT) {
			return new TransformedCursor_RA(this, null, outT, weightT);
		}
		public default RandomAccess transformOutputs(CursorTransformer inT, CursorTransformer outT) {
			return new TransformedCursor_RA(this, inT, outT, null);
		}
		public default RandomAccess transformOutputs(CursorTransformer inT, CursorTransformer outT, DoubleUnaryOperator weightT) {
			return new TransformedCursor_RA(this, inT, outT, weightT);
		}
		public default RandomAccess transformWeights(DoubleUnaryOperator weightT) {
			return new TransformedCursor_RA(this, null, null, weightT);
		}
		
		
	}
	default TrainingBatchCursor remap(double weightScale, int[] outputMapping, int... inputMapping) {
		return new RemappedCursor<TrainingBatchCursor>(this, weightScale, outputMapping, inputMapping);
	}
	default TrainingBatchCursor scaleWeights(double weightScale) {
		return new RemappedCursor<TrainingBatchCursor>(this, weightScale, null, null);
	}
	
	
	///////////////////////static factory methods
	/**
	 * Concatenate several sequences. 
	 * The sequences must have the same format (# of input and output variables).
	 * Even though I can't imagine a use case, you can concatenate a sequence with itself
	 * @param sequences
	 * @return
	 */
	public static TrainingBatchCursor concat(TrainingBatchCursor... sequences) {
		if(sequences.length==1)
			return sequences[0];
		TrainingBatchCursor ret = new TrainingBatchCursor_Concat<TrainingBatchCursor>(sequences);
		ret.reset();
		return ret;
	}
	/**
	 * Concatenate several sequences. 
	 * The sequences must have the same format (# of input and output variables).
	 * Even though I can't imagine a use case, you can concatenate a sequence with itself
	 * @param sequences
	 * @return
	 */
	public static TrainingBatchCursor concat(List<? extends TrainingBatchCursor> sequences) {
		return concat(sequences.toArray(new TrainingBatchCursor[sequences.size()]));
	}
	/**
	 * Concatenate several sequences. 
	 * The sequences must have the same format (# of input and output variables).
	 * Even though I can't imagine a use case, you can concatenate a sequence with itself
	 * @param sequences
	 * @return
	 */
	public static TrainingBatchCursor.RandomAccess concatRandomAccess(List<? extends TrainingBatchCursor.RandomAccess> sequences) {
		return concat(sequences.toArray(new TrainingBatchCursor.RandomAccess[sequences.size()]));
	}
	/**
	 * Concatenate several sequences. 
	 * The sequences must have the same format (# of input and output variables).
	 * Even though I can't imagine a use case, you can concatenate a sequence with itself
	 * @param sequences
	 * @return
	 */
	public static RandomAccess concat(RandomAccess... sequences) {
		if(sequences.length==1)
			return sequences[0];
		class TrainingBatchCursor_ConcatRandomAccess 
		extends TrainingBatchCursor_Concat<RandomAccess> implements RandomAccess{
			long[] index;
			public TrainingBatchCursor_ConcatRandomAccess(RandomAccess[] sequences) {
				super(sequences);
				index=new long[sequences.length+1];
				for(int i=0; i<sequences.length; ++i)
					index[i+1]=index[i]+sequences[i].size();
			}
			@Override
			public long size() {
				return index[sequences.length];
			}
			@Override
			public void seek(long position) {
				if(position<0)
					throw new IllegalArgumentException("Seek position must mut be negative");
				int seqi=seqi(position);
				if(seqi>=sequences.length) {
					seqi=sequences.length-1;
					position=index[sequences.length];
				}
				currentSequenceIndex=seqi;
				sequences[currentSequenceIndex].seek(position-index[seqi]);
				ff();
				
			}
			private int seqi(long position) {
				int ret = Arrays.binarySearch(index, position);
				if(ret>=0)
					return ret;
				return -ret-2;
			}
			@Override
			public RandomAccess clone() {
				RandomAccess[] sc = sequences.clone();
				for(int i=0; i<sequences.length; ++i) {
					sc[i]=sequences[i].clone();
				}
				return concat(sc);
			}
			@Override
			public RandomAccess subsequence(long start, long end) {
				if(start<0)
					start=0;
				if(end>size())
					end=size();
				if(end<=start)
					return empty(inputCount(), outputCount());
				ArrayList<RandomAccess> sv = new ArrayList<>();
				for(int i=seqi(start); i<sequences.length; ++i) {
					if(index[i+1]<=start)
						continue;
					if(index[i]<start) {
						sv.add(sequences[i].subsequence(start-index[i], end-index[i]));
						continue;
					}
					if(index[i]>=end)
						break;
					if(index[i+1]>end) {
						sv.add(sequences[i].subsequence(0, end-index[i]));
						break;
					}
					sv.add(sequences[i].clone());
				}
				return concat(sv.toArray(new RandomAccess[sv.size()]));
				
			}
			
		}
		RandomAccess ret = new TrainingBatchCursor_ConcatRandomAccess(sequences);
		ret.reset();
		return ret;
	}
	public static TrainingBatchCursor.RandomAccess singleton(double[] inputs, double[] outputs, double weight) {
		return new TrainingBatchCursor.RandomAccess() {
			@Override public int outputCount() {return outputs.length;}
			@Override public int inputCount() {return inputs.length;}
			@Override public void giveOutputs(double[] outputs0) {
				System.arraycopy(outputs, 0, outputs0, 0, outputs.length);
			}
			@Override public void giveInputs(double[] inputs0) {
				System.arraycopy(inputs, 0, inputs0, 0, inputs.length);
			}
			@Override public double getWeight() {return weight;}
			boolean consumed=false;
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
			@Override
			public RandomAccess clone() {
				return singleton(inputs, outputs, weight);
			}
			@Override
			public long size() {
				return 1;
			}
			@Override
			public void seek(long position) {
				consumed=position>0;
			}
		};
	}
	public static RandomAccess singleton(double[] inputs, double output, double weight) {
		return new TrainingBatchCursor.RandomAccess() {
			@Override public int outputCount() {return 1;}
			@Override public int inputCount() {return inputs.length;}
			@Override public void giveOutputs(double[] outputs0) {
				outputs0[0]=output;
			}
			@Override public void giveInputs(double[] inputs0) {
				System.arraycopy(inputs, 0, inputs0, 0, inputs.length);
			}
			@Override public double getWeight() {return weight;}
			boolean consumed=false;
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
			@Override
			public RandomAccess clone() {
				return singleton(inputs, output, weight);
			}
			@Override
			public long size() {
				return 1;
			}
			@Override
			public void seek(long position) {
				consumed=position>0;
			}
		};
	}
	public static RandomAccess empty(int inputCount, int outputCount) {
		return new RandomAccess() {
			@Override public RandomAccess clone(){return empty(inputCount, outputCount);}
			@Override public double getWeight() {throw new UnsupportedOperationException("Empty cursor");}
			@Override public void giveInputs(double[] inputs) {throw new UnsupportedOperationException("Empty cursor");}
			@Override public void giveOutputs(double[] outputs) {throw new UnsupportedOperationException("Empty cursor");}
			@Override public int inputCount() {return inputCount;}
			@Override public int outputCount() {return outputCount;}
			@Override public boolean isFinished() {return true;}
			@Override public void next() {throw new UnsupportedOperationException("Empty cursor");}
			@Override public void reset() {}
			@Override public void seek(long position) {}
			@Override public long size() {return 0;}
			@Override public RandomAccess subsequence(long start, long end) {return this;}
		};
	}
	public static class TrainingBatchCursor_Concat<B extends TrainingBatchCursor> implements TrainingBatchCursor {
		protected final int outDim;
		protected final int inDim;
		protected final B[] sequences;
		protected int currentSequenceIndex = 0;
	
		public TrainingBatchCursor_Concat(B[] sequences) {
			if(sequences.length==0) {
				throw new IllegalArgumentException("Must concatenate a nonzero number of sequences");
			}
			int inDim = sequences[0].inputCount();
			int outDim = sequences[0].outputCount();
			for(int i=1; i<sequences.length; ++i) {
				TrainingBatchCursor it = sequences[i];
				if(it.inputCount()!=inDim)
					throw new IllegalArgumentException("Sequence #"+i+" has a different number of input variables than sequence #0");
				if(it.outputCount()!=outDim)
					throw new IllegalArgumentException("Sequence #"+i+" has a different number of output variables than sequence #0");
			}
			this.outDim = outDim;
			this.inDim = inDim;
			this.sequences = sequences;
		}
	
		@Override public int outputCount() {return outDim;}
	
		@Override public int inputCount() {return inDim;}
	
		@Override
		public void giveOutputs(double[] outputs) {
			if(isFinished())
				throw new IllegalStateException("This cursor has reached its end.");
			sequences[currentSequenceIndex].giveOutputs(outputs);
		}
	
		@Override
		public void giveInputs(double[] inputs) {
			if(isFinished())
				throw new IllegalStateException("This cursor has reached its end.");
			sequences[currentSequenceIndex].giveInputs(inputs);
		}
	
		@Override
		public double getWeight() {
			if(isFinished())
				throw new IllegalStateException("This cursor has reached its end.");
			return sequences[currentSequenceIndex].getWeight();
		}
	
		@Override
		public void reset() {
			sequences[0].reset();
			currentSequenceIndex=0;
			ff();
		}
	
		protected void ff() {
			while(currentSequenceIndex<sequences.length && sequences[currentSequenceIndex].isFinished()) {
				++currentSequenceIndex;
				if(currentSequenceIndex>=sequences.length)
					break;
				sequences[currentSequenceIndex].reset();
			}
		}
	
		@Override
		public void next() {
			if(isFinished())				
				throw new IllegalStateException("This cursor has reached its end.");
			sequences[currentSequenceIndex].next();
			ff();
		}
	
		@Override
		public boolean isFinished() {
			ff();
			return currentSequenceIndex>=sequences.length;
		}
	
		@SuppressWarnings("unchecked")
		@Override
		public TrainingBatchCursor clone() {
			B[] sc = sequences.clone();
			for(int i=0; i<sequences.length; ++i) {
				sc[i]=(B) ((TrainingBatchCursor)sequences[i]).clone();
			}
			return concat(sc);
		}
	}

	static class PolyInteractionCursor_RA extends PolyInteractionCursor<RandomAccess> implements RandomAccess{

		public PolyInteractionCursor_RA(RandomAccess back, int order, int prefixLength, int suffixLength) {
			super(back, order, prefixLength, suffixLength);
		}

		@Override
		public long size() {
			return this.back.size();
		}

		@Override
		public void seek(long position) {
			this.back.seek(position);
		}

		@Override
		public PolyInteractionCursor_RA clone() {
			return new PolyInteractionCursor_RA(this.back.clone(), this.order, this.prefixLength, this.suffixLength);
		}
		
	}
	
	
	static class TransformedCursor_RA extends TransformedCursor<RandomAccess> implements RandomAccess{

		public TransformedCursor_RA(RandomAccess back, CursorTransformer inputTransformer, CursorTransformer outputTransformer, DoubleUnaryOperator weightTransformer) {
			super(back, inputTransformer, outputTransformer, weightTransformer);
		}

		@Override
		public long size() {
			return this.back.size();
		}

		@Override
		public void seek(long position) {
			this.back.seek(position);
		}

		@Override
		public TransformedCursor_RA clone() {
			return new TransformedCursor_RA(this.back.clone(), inputTransform, outputTransform, weightTransform);
		}
		
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
		if(order==1)
			return back;
		return new PolyInteractionCursor<TrainingBatchCursor>(back, order, 0, 0);
	}
	/**
	 * Make a wrapper for a {@link TrainingBatchCursor} that enriches its inputs by
	 * taking polynomial combinations of the original data's inputs up to a given order
	 * 
	 * @param back
	 * @param order
	 * @return
	 */
	public static TrainingBatchCursor.RandomAccess polyInteractionFeatures(
			TrainingBatchCursor.RandomAccess back, int order) {
		if(order==1)
			return back;
		return new PolyInteractionCursor_RA(back, order, 0, 0);
	}
	/**
	 * Make a wrapper for a {@link TrainingBatchCursor} that enriches its inputs by
	 * taking polynomial combinations of the original data's inputs up to a given order
	 * 
	 * @param back
	 * @param order
	 * @return
	 */
	public static TrainingBatchCursor polyInteractionFeatures(TrainingBatchCursor back, int order, int prefixLength, int suffixLength) {
		if(order==1)
			return back;
		return new PolyInteractionCursor<TrainingBatchCursor>(back, order, prefixLength, suffixLength);
	}
	/**
	 * Make a wrapper for a {@link TrainingBatchCursor} that enriches its inputs by
	 * taking polynomial combinations of the original data's inputs up to a given order
	 * 
	 * @param back
	 * @param order
	 * @return
	 */
	public static TrainingBatchCursor.RandomAccess polyInteractionFeatures(
			TrainingBatchCursor.RandomAccess back, int order, int prefixLength, int suffixLength) {
		if(order==1)
			return back;
		return new PolyInteractionCursor_RA(back, order, prefixLength, suffixLength);
	}
	
	public default RamBuffer ramBuffer() {
		return ramBuffer(this);
	}
	public static RamBuffer ramBuffer(TrainingBatchCursor of) {
		if(of instanceof RamBuffer)
			return ((RamBuffer)of).clone();
		ArrayList<double[]> inputs=new ArrayList<>(); 
		ArrayList<double[]> outputs=new ArrayList<>(); 
		ArrayList<Double> weights=new ArrayList<>();
		int ic = of.inputCount();
		int oc = of.outputCount();
		of.reset();
		long count = 0;
		while(!of.isFinished()) {
			double[] i = new double[ic];
			double[] o = new double[oc];
			of.giveInputs(i);
			of.giveOutputs(o);
			inputs.add(i);
			outputs.add(o);
			weights.add(of.getWeight());
			of.next();
			if(count>=Integer.MAX_VALUE) {
				throw new ArrayIndexOutOfBoundsException();
			}
			++count;
		}
		double[][] is = inputs.toArray(new double[inputs.size()][]);
		double[][] os = outputs.toArray(new double[outputs.size()][]);
		double[] ws = new double[weights.size()];
		for(int i=0; i<ws.length; ++i)
			ws[i]=weights.get(i);

		return new RamBuffer(is, os, ws, 0, ic, oc);
	}

	public static class RamBuffer implements RandomAccess{
		RamBuffer(double[][] is2, double[][] os2, double[] ws2, int at2, int ic, int oc) {
			
			is=is2;
			os=os2;
			ws=ws2;
			at=at2;
			this.ic = ic;
			this.oc = oc;
		}
		final int ic, oc;
		final double[][] is, os;
		final double[] ws;
		int at;
		@Override
		public void next() {
			++at;
		}
		@Override
		public boolean isFinished() {
//			if(is.length==0)
//				System.out.println();
			return at>=is.length;
		}
		@Override
		public void reset() {
			at=0;
		}
		@Override
		public int inputCount() {
			return ic;
		}
		@Override
		public int outputCount() {
			return oc;
		}
		@Override
		public void giveInputs(double[] inputs) {
			System.arraycopy(is[at], 0, inputs, 0, ic);
		}
		@Override
		public void giveOutputs(double[] outputs) {
			System.arraycopy(os[at], 0, outputs, 0, oc);
		}
		@Override
		public double getWeight() {
			return ws==null?1:ws[at];
		}
		@Override
		public long size() {
			return is.length;
		}
		@Override
		public void seek(long position) {
			if(position<0 || position>=is.length)
				throw new IndexOutOfBoundsException(String.valueOf(position));
			at=(int) position;
			
		}
		@Override
		public RamBuffer clone() {
			return new RamBuffer(is, os, ws, at, ic, oc);
		}
		/**
		 * Shuffles the data in this RamBuffer and all its clones
		 * @return
		 */
		public RamBuffer shuffle(Random rnd) {
			int l = is.length;
			for(int i=l-1; i>0; i--){
				int sel=rnd.nextInt(i+1);
				if(sel==i)
					continue;
				{
					double[] t=is[i];
					is[i]=is[sel];
					is[sel]=t;
				}
				{
					double[] t=os[i];
					os[i]=os[sel];
					os[sel]=t;
				}
				{
					double t=ws[i];
					ws[i]=ws[sel];
					ws[sel]=t;
				}
			}
			return this;
		}
		
	}
	@Override
	default TrainingBatchCursor whitened(Whitener whitenInputs, Whitener whitenOutputs) {
		return new WhitenedTrainingBatchCursor<TrainingBatchCursor>(this, whitenInputs, whitenOutputs);
	}
	static public class WhitenedTrainingBatchCursor<B extends TrainingBatchCursor> 
	extends WhitenedTrainingDataGiver<B>
	implements TrainingBatchCursor{

		public WhitenedTrainingBatchCursor(B back, Whitener whitenInputs, Whitener whitenOutputs) {
			super(back, whitenInputs, whitenOutputs);
		}

		@Override
		public void next() {
			back.next();
		}

		@Override
		public boolean isFinished() {
			return back.isFinished();
		}

		@Override
		public void reset() {
			back.reset();
		}

		@SuppressWarnings("unchecked")
		@Override
		public WhitenedTrainingBatchCursor<B> clone() {
			return new WhitenedTrainingBatchCursor<B>((B)((TrainingBatchCursor)back).clone(), whitenInputs, whitenOutputs);
		}
		
	}
	public static class RemappedCursor<B extends TrainingBatchCursor> implements TrainingBatchCursor{
		int[] inputMapping;
		int[] outputMapping;
		public final B original;
		final int ic;
		final int oc;
		final double[] buffer;
		final double weightScale;
		@SuppressWarnings("unchecked")
		@Override
		public RemappedCursor<B> clone() {
			return new RemappedCursor<B>((B)((TrainingBatchCursor)original).clone(), weightScale, outputMapping, inputMapping);
		}
		public RemappedCursor(B back, double weightScale, int[] outputMapping, int...inputMapping) {
			original = back;
			this.inputMapping = inputMapping;
			ic = inputMapping==null?back.inputCount():inputMapping.length;
			this.outputMapping = outputMapping;
			oc = outputMapping==null?back.outputCount():outputMapping.length;
			buffer = new double[Math.max(back.inputCount(), back.outputCount())];
			this.weightScale=weightScale;
		}
		@Override public int inputCount() {return ic;}
		@Override public int outputCount() {return oc;}
		@Override public void giveInputs(double[] inputs) {
			if(inputMapping==null) 
				original.giveInputs(inputs);
			else {
				original.giveInputs(buffer);
				for(int i=0; i<ic; ++i) {
					inputs[i]=buffer[inputMapping[i]];
				}
			}
		}

		@Override
		public void giveOutputs(double[] outputs) {
			if(outputMapping==null) 
				original.giveOutputs(outputs);
			else {
				original.giveOutputs(buffer);
				for(int i=0; i<ic; ++i) {
					outputs[i]=buffer[outputMapping[i]];
				}
			}			
		}
		@Override public double getWeight() {return original.getWeight()*weightScale;}
		@Override public void next() {original.next();}
		@Override public boolean isFinished() {return original.isFinished();}
		@Override public void reset() {original.reset();}
		static class RandomAccess extends RemappedCursor<TrainingBatchCursor.RandomAccess> implements TrainingBatchCursor.RandomAccess{

			public RandomAccess(TrainingBatchCursor.RandomAccess back, double weightScale, int[] outputMapping, int... inputMapping) {
				super(back, weightScale, outputMapping, inputMapping);
			}

			@Override public long size() {return original.size();}
			@Override public void seek(long position) {original.seek(position);}
			@Override public RemappedCursor.RandomAccess clone() {
				return new RemappedCursor.RandomAccess(original, weightScale, outputMapping, inputMapping);
			}
			
		}
	}
	public default double getTotalWeight() {
		reset();
		double ret = 0;
		while(!isFinished()) {
			ret += getWeight();
			next();
		}
		return ret;
	}
	public default boolean containsNaN() {
		reset();
		double[] in = new double[inputCount()];
		double[] out = new double[outputCount()];
		while(!isFinished()) {
			double w = getWeight();
			if(Double.isNaN(w))
				return true;
			giveInputs(in);
			giveOutputs(out);
			for(double d: in)
				if(Double.isNaN(d))
					return true;
			for(double d: out)
				if(Double.isNaN(d))
					return true;
			next();
		}
		return false;
	}
	
	public default TrainingBatchCursor transformInputs(CursorTransformer inT) {
		return new TransformedCursor<>(this, inT, null, null);
	}
	public default TrainingBatchCursor transformInputs(CursorTransformer inT, DoubleUnaryOperator weightT) {
		return new TransformedCursor<>(this, inT, null, weightT);
	}
	public default TrainingBatchCursor transformOutputs(CursorTransformer outT) {
		return new TransformedCursor<>(this, null, outT, null);
	}
	public default TrainingBatchCursor transformOutputs(CursorTransformer outT, DoubleUnaryOperator weightT) {
		return new TransformedCursor<>(this, null, outT, weightT);
	}
	public default TrainingBatchCursor transformOutputs(CursorTransformer inT, CursorTransformer outT) {
		return new TransformedCursor<>(this, inT, outT, null);
	}
	public default TrainingBatchCursor transformOutputs(CursorTransformer inT, CursorTransformer outT, DoubleUnaryOperator weightT) {
		return new TransformedCursor<>(this, inT, outT, weightT);
	}
	public default TrainingBatchCursor transformWeights(DoubleUnaryOperator weightT) {
		return new TransformedCursor<>(this, null, null, weightT);
	}

    
	
}
