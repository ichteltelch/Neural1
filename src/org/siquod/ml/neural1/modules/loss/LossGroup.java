package org.siquod.ml.neural1.modules.loss;

import java.util.ArrayList;

/**
 * {@link LossGroup}s allow you to interpret the output vector as several probability 
 * distributions stored in sequence, and to compute their softmax anf NLL losses separately.
 * A {@link LossGroup} object specifies which outputs belong together and how to treat them.
 * If there is only one output in a {@link LossGroup}, it is taken to specify a single probability
 * p of a Bernoulli distribution, with the other probability being implicitly (1-p)
 * @author bb
 *
 */
final public class LossGroup{
	/**
	 * The index of the first output belonging to the group 
	 */
	public final int start;
	/**
	 * One plus the index of the last output belonging to the group
	 */
	public final int end;
	/**
	 * The number of outputs in the group
	 */
	public final int length;
	/**
	 * An index used for gating the loss: The loss contributions from this group are
	 * multiplied with the value of the target distribution at the specified index, 
	 * if it is non-negative, but see also {@link #gateInverted}
	 * 
	 */
	public final int gate;
	/**
	 * If <code>true</code>, the target value p at the {@link #gateInverted} will be inverted to 1-p
	 */
	public final boolean gateInverted;
	/**
	 * Multiply the loss contributions by this factor
	 */
	public final float weight;
	public LossGroup(int start, int length) {
		this(start, length, -1, false, 1);
	}
	public LossGroup(int start, int length, int gate, boolean gateInverted, float weight) {
		this.start = start;
		this.length = length;
		this.end = start + length;
		this.gate = gate;
		this.gateInverted = gateInverted;
		this.weight = weight;
	}
	public boolean isGated() {
		return gate>=0;
	}
	public boolean isSingleton() {
		return length==1;
	}
	/**
	 * @return A fresh {@link LossGroupsBuilder}
	 */
	public static LossGroupsBuilder b() {return new LossGroupsBuilder();}
	/**
	 * Make an array containing a single ungated {@link LossGroup} with {@link #weight} 1.
	 * @param count
	 * @return
	 */
	public static LossGroup[] makeDefault(int count) {
		return new LossGroup[] {new LossGroup(0, count)};
	}
	public static final class LossGroupsBuilder{
		ArrayList<LossGroup> groups=new ArrayList<>();
		int position=0;
		int gate=-1;
		boolean inverted=false;
		float weight = 1;
		/**
		 * Add a singleton group for a Bernoulli distribution
		 * @return
		 */
		public LossGroupsBuilder singleton() {
			groups.add(new LossGroup(position++, 1, gate, inverted, weight));
			return this;
		}
		/**
		 * Add a group of the specified length
		 * @param length
		 * @return
		 */
		public LossGroupsBuilder group(int length) {
			groups.add(new LossGroup(position+=length, length, gate, inverted, weight));
			return this;
		}
		/**
		 * Set the weighting factor for the following groups
		 * @param weight
		 * @return
		 */
		public LossGroupsBuilder weight(float weight) {
			this.weight=weight;
			return this;
		}
		/**
		 * Set the gating index for the following groups.
		 * The gate will be not inverted
		 * @param index
		 * @return
		 */
		public LossGroupsBuilder gate(int index) {
			gate = index;
			inverted = false;
			return this;
		}
		/**
		 * Set the gating index for the following groups,
		 * and whether the gate will be inverted
		 * @param index
		 * @param inverted
		 * @return
		 */
		public LossGroupsBuilder gate(int index, boolean inverted) {
			gate = index;
			this.inverted=inverted;
			return this;
		}
		/**
		 * Change whether the gate is inverted for the following groups
		 * @param inverted
		 * @return
		 */
		public LossGroupsBuilder gateInverted(boolean inverted) {
			this.inverted=inverted;
			return this;
		}
		/**
		 * Don't use gating for the following groups
		 * @param index
		 * @return
		 */
		public LossGroupsBuilder ungated() {
			gate = -1;
			return this;
		}
		/**
		 * 
		 * @return How many indices have currently been specified; the {@link LossGroup#start}
		 * index the next specified {@link LossGroup} would have  
		 */
		int totalLength() { 
			return position;
		}
		/**
		 * assert that {@link #totalLength()} is the given value
		 * @param length
		 * @return
		 */
		public LossGroupsBuilder assertTotalLength(int length) {
			assert length==position;
			return this;
		}
		/**
		 * Get an array of the {@link LossGroup}s specified so far
		 * @return
		 */
		public LossGroup[] toArray() {
			return groups.toArray(new LossGroup[groups.size()]);
		}
	}
}








