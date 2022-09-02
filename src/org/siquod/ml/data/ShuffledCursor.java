package org.siquod.ml.data;

import java.util.Random;

public class ShuffledCursor implements TrainingBatchCursor.RandomAccess {
	final TrainingBatchCursor.RandomAccess back;
	int[] permutation;
	int index;
	Random rand;
	boolean fresh;
	public ShuffledCursor(RandomAccess back, Random rand) {
		this.back=back;
		index=0;
		this.rand=DataManagement.cloneRandom(rand);
		initPermutation();	
	}
	public ShuffledCursor(RandomAccess back) {
		this(back, new Random());
	}
	private void initPermutation() {
		if(back.size()>Integer.MAX_VALUE)
			throw new IllegalStateException("Sequence is too long to be shuffled");
		permutation=new int[(int)back.size()];
		for(int i=0; i<permutation.length; ++i)
			permutation[i]=i;
		shuffle(permutation);
		back.seek(permutation[index]);
	}
	public void shuffle(int[] data) {
		for(int i=0; i<data.length; ++i) {
			int j = i + rand.nextInt(data.length-i);
			if(i!=j) {
				int t = data[i];
				data[i]=data[j];
				data[j]=t;
			}
		}
	}
	private ShuffledCursor(RandomAccess back, int[] permutation, int index, Random rand) {
		this.back=back;
		this.permutation=permutation;
		this.index=index;
		this.rand=rand;
	}
	@Override
	public ShuffledCursor clone() {
		return new ShuffledCursor(back.clone(), permutation.clone(), index, DataManagement.cloneRandom(rand));
	}
	@Override
	public double getWeight() {
		return back.getWeight();
	}
	@Override
	public void giveInputs(double[] inputs) {
		back.giveInputs(inputs);
	}
	@Override
	public void giveOutputs(double[] outputs) {
		back.giveOutputs(outputs);
	}
	@Override
	public int inputCount() {
		return back.inputCount();
	}
	@Override
	public boolean isFinished() {
		return index>=permutation.length;
	}
	@Override
	public void next() {
		index++;
		if(index<permutation.length)
			back.seek(permutation[index]);
	}
	@Override
	public int outputCount() {
		return back.outputCount();
	}
	@Override
	public void reset() {
		index=0;
		fresh=false;
		back.seek(permutation[index]);
	}
	@Override
	public void seek(long position) {
		if(position<0 || position>=permutation.length)
			throw new IndexOutOfBoundsException(String.valueOf(position));
		index=(int)position;
		back.seek(permutation[index]);

	}
	@Override
	public long size() {
		return permutation.length;
	}
	public ShuffledCursor shuffle() {
		index=0;
		if(back.size()!=permutation.length)
			initPermutation();
		else {
			if(!fresh || index>0)
				shuffle(permutation);
		}
		fresh=true;
		back.seek(permutation[index]);

		
		return this;
	}

}
