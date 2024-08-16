package org.siquod.ml.neural1.modules;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.siquod.ml.neural1.ActivationBatch;
import org.siquod.ml.neural1.ActivationSeq;
import org.siquod.ml.neural1.ActivationSet;
import org.siquod.ml.neural1.ForwardPhase;
import org.siquod.ml.neural1.Interface;
import org.siquod.ml.neural1.InterfaceAllocator;
import org.siquod.ml.neural1.Module;
import org.siquod.ml.neural1.ParamAllocator;
import org.siquod.ml.neural1.ParamBlocks;
import org.siquod.ml.neural1.ParamSet;
import org.siquod.ml.neural1.TensorFormat;

/**
 * This module performs a linear mapping from its input vector to its output vector
 * @author bb
 *
 */
public class QuadraticInteraction implements InOutModule{

	Interface in;
	Interface out;
	InOutFactory leftFactory;
	InOutFactory rightFactory;
	InOutFactory afterFactory;
	InOutModule leftModule;
	InOutModule rightModule;
	InOutModule afterModule;
	Interface leftFactors;
	Interface rightFactors;
	Interface products;
	TensorFormat in2d;
	TensorFormat product2d;
	TensorFormat left2d;
	TensorFormat right2d;
	Module[] exec;
	Kernel[] kernels;
	int leftLength;
	int rightLength;
	int outLength;
	public QuadraticInteraction(QuadraticInteraction copyThis) {
		this.in = copyThis.in;
		this.out = copyThis.out;
		this.leftFactory = copyThis.leftFactory;
		this.rightFactory = copyThis.rightFactory;
		this.afterFactory = copyThis.afterFactory;
		this.leftModule = copyThis.leftModule.copy();
		this.rightModule = copyThis.rightModule.copy();
		this.afterModule = copyThis.afterModule.copy();
		this.leftFactors = copyThis.leftFactors;
		this.rightFactors = copyThis.rightFactors;
		this.products = copyThis.products;
		this.in2d = copyThis.in2d;
		this.product2d = copyThis.product2d;
		this.left2d = copyThis.left2d;
		this.right2d = copyThis.right2d;
		makeExec();
		this.kernels= copyThis.kernels;
		this.leftLength= copyThis.leftLength;
		this.rightLength= copyThis.rightLength;
		this.outLength= copyThis.outLength;
	}
	public QuadraticInteraction copy() {
		return new QuadraticInteraction(this);
	}
	public static QuadraticInteractionBuilder b() {
		return new QuadraticInteractionBuilder();
	}
	public static class Kernel{
		public final int leftDim;
		public final int midDim;
		public final int rightDim;
		public final int repetitions;
		public final int leftMatDim;
		public final int rightMatDim;
		public final int outMatDim;
		public final int leftStart;
		public final int rightStart;
		public final int outStart;
		public final int leftLength;
		public final int rightLength;
		public final int outLength;
		public final int leftEnd;
		public final int rightEnd;
		public final int outEnd;
		public final boolean symmetric;
		public final float scaleDown;

		public Kernel(int leftDim, int midDim, int rightDim, int repetitions, int leftStart, int rightStart, int outStart, float scaleDown) {
			this.leftDim = leftDim;
			this.rightDim = rightDim;
			this.midDim = midDim;
			this.repetitions = repetitions;
			this.leftStart = leftStart;
			this.rightStart = rightStart;
			this.outStart = outStart;
			symmetric = false;
			leftMatDim = leftDim * midDim;
			rightMatDim = midDim * rightDim;
			outMatDim = leftDim * rightDim;
			leftLength = leftMatDim * repetitions;
			rightLength = rightMatDim * repetitions;
			outLength = outMatDim * repetitions;
			leftEnd = leftStart + leftLength;
			rightEnd = rightStart + rightLength;
			outEnd = outStart + outLength;
			this.scaleDown=scaleDown;
		}
		public Kernel(int outerDim, int innerDim, int repetitions, int leftStart, int rightStart, int outStart, float scaleDown) {
			this.leftDim = outerDim;
			this.rightDim = outerDim;
			this.midDim = innerDim;
			this.repetitions = repetitions;
			this.leftStart = leftStart;
			this.rightStart = rightStart;
			this.outStart = outStart;
			symmetric = true;
			leftMatDim = outerDim * innerDim;
			rightMatDim = 0;
			outMatDim = (outerDim * (outerDim+1)) /2;
			leftLength = leftMatDim * repetitions;
			rightLength = 0;
			outLength = outMatDim * repetitions;
			leftEnd = leftStart + leftLength;
			rightEnd = rightStart + rightLength;
			outEnd = outStart + outLength;
			this.scaleDown=scaleDown;
		}
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append('[').append(repetitions).append(',').append(leftDim).append(',').append(midDim);
			if(!symmetric)
				sb.append(',').append(rightDim);
			sb.append(']');
			return sb.toString();
		}
		public int[] toIntArray() {
			if(symmetric)
				return new int[] {repetitions, leftDim, midDim};
			else 
				return new int[] {repetitions, leftDim, midDim, rightDim};
		}
	}



	@Override
	public void allocate(InterfaceAllocator ia) {
		in=ia.get("in");
		out=ia.get("out");
		if(!in.tf.equalExceptChannels(out.tf))
			throw new IllegalArgumentException("Input and output dimensions (except last one) must be equal");
		leftFactors = ia.allocate(new Interface("left", in.tf.withChangedChannels(leftLength)));
		rightFactors = ia.allocate(new Interface("right", in.tf.withChangedChannels(rightLength)));
		products = ia.allocate(new Interface("products", in.tf.withChangedChannels(in.tf.channels()+outLength)));

		in2d = in.tf.to2D();
		product2d = products.tf.to2D();
		left2d = leftFactors.tf.to2D();
		right2d = rightFactors.tf.to2D();

		leftModule = leftFactory.produce(in, leftFactors);
		rightModule = rightFactory.produce(in, rightFactors);
		afterModule = afterFactory.produce(products, out);

		leftModule.allocate(ia, "in", "left");
		rightModule.allocate(ia, "in", "right");
		afterModule.allocate(ia, "products", "out");

		makeExec();
	}
	private void makeExec() {
		exec = new Module[] {
				leftModule, rightModule, afterModule,
		};
	}

	QuadraticInteraction(Kernel[] kernels, 
			InOutFactory left, InOutFactory right, InOutFactory after,  
			int leftLength, int rightLength, int outLength){
		assert kernels != null;
		assert kernels.length>0;
		Kernel lastKernel = kernels[kernels.length-1];
		assert leftLength == lastKernel.leftEnd;
		assert rightLength == lastKernel.rightEnd;
		assert outLength == lastKernel.outEnd;
		this.leftLength = leftLength;
		this.rightLength = rightLength;
		this.outLength = outLength;
		this.kernels=kernels;
		leftFactory = left;
		rightFactory = right;
		afterFactory =after;

	}


	@Override
	public void allocate(ParamAllocator ia) {
		ia.push(null); leftModule.allocate(ia); ia.pop();
		ia.push(null); rightModule.allocate(ia); ia.pop();
		ia.push(null); afterModule.allocate(ia); ia.pop();
	}

	@Override
	public void share(ParamBlocks ps) {
		leftModule.share(ps.get("left"));
		rightModule.share(ps.get("right"));
		afterModule.share(ps.get("after"));
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret=new ParamBlocks("QI");
		ret.add("left", leftModule.getParamBlocks());
		ret.add("right", rightModule.getParamBlocks());
		ret.add("after", afterModule.getParamBlocks());
		return ret;	
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		leftModule.forward(training, params, as, t, inst);
		rightModule.forward(training, params, as, t, inst);
		if(inst==null) {
			final int startA = 0;
			final int endA = as.length;
			final int startBri = 0;
			final int endBri = in2d.dims[0];
			final int inCount = in2d.channels();
			for(int ai = startA; ai<endA; ++ai) {
				final ActivationSeq at = as.a[ai];
				if(at==null) continue;
				final ActivationSet a = at.get(t);
				for(int bri=startBri; bri<endBri; ++bri) {
					//Copy unmodified inputs
					for(int i=0; i<inCount; ++i) {
						a.add(products, product2d.index(bri, i), a.get(in, in2d.index(bri, i)));
					}
					//Compute quadratic interactions
					for(Kernel krn: kernels) {
						for(int rep=0; rep<krn.repetitions; ++rep) {
							final int leftOffset = krn.leftStart + rep*krn.leftMatDim;
							final int rightOffset = krn.rightStart + rep*krn.rightMatDim;
							final int outOffset = inCount + krn.outStart + rep*krn.outMatDim;
							for(int l = 0, oi=outOffset; l<krn.leftDim; ++l) {
								final int endR = krn.symmetric?l+1:krn.rightDim;
								for(int r = 0; r<endR; ++r, ++oi) {
									float sum = 0;
									for(int m=0; m<krn.midDim; ++m) {
										final int li = leftOffset + l*krn.midDim + m;
										final float fl = a.get(leftFactors, left2d.index(bri, li));
										final float fr;
										if(krn.symmetric) {
											final int ri = leftOffset + r*krn.midDim + m;
											fr = a.get(leftFactors, left2d.index(bri, ri));
										}else {
											final int ri = rightOffset + r*krn.midDim + m;
											fr = a.get(rightFactors, right2d.index(bri, ri));
										}
										sum += fl*fr;
									}
									a.add(products, product2d.index(bri, oi), sum*krn.scaleDown);
								}
							}
						}
					}

				}
			}

		}else {
			int startA = 0;
			int endA = as.length;
			int inCount = in.channels();
			for(int ai = startA; ai<endA; ++ai) {
				ActivationSeq at = as.a[ai];
				if(at==null) continue;
				ActivationSet a = at.get(t);
				{
					//Copy unmodified inputs
					for(int i=0; i<inCount; ++i) {
						a.add(products, inst, i, a.get(in, inst, i));
					}
					//Compute quadratic interactions
					for(Kernel krn: kernels) {
						for(int rep=0; rep<krn.repetitions; ++rep) {
							int leftOffset = krn.leftStart + rep*krn.leftMatDim;
							int rightOffset = krn.rightStart + rep*krn.rightMatDim;
							int outOffset = inCount + krn.outStart + rep*krn.outMatDim;
							for(int l = 0, oi=outOffset; l<krn.leftDim; ++l) {
								int endR = krn.symmetric?l+1:krn.rightDim;
								for(int r = 0; r<endR; ++r, ++oi) {
									float sum = 0;
									for(int m=0; m<krn.midDim; ++m) {
										int li = leftOffset + l*krn.midDim + m;
										float fl = a.get(leftFactors, inst, li);
										float fr;
										if(krn.symmetric) {
											int ri = leftOffset + r*krn.midDim + m;
											fr = a.get(leftFactors, inst, ri);
										}else {
											int ri = rightOffset + r*krn.midDim + m;
											fr = a.get(rightFactors, inst, ri);
										}
										sum += fl*fr;
									}
									a.add(products, inst, oi, sum*krn.scaleDown);
								}
							}
						}
					}

				}
			}
		}
		afterModule.forward(training, params, as, t, inst); 
	}



	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {	
		if(dontBackprop.contains(phase))
			return;
		afterModule.backprop(phase, params, as, errors, t, inst);
		if(inst==null) {
			final int startA = 0;
			final int endA = as.length;
			final int startBri = 0;
			final int endBri = in2d.dims[0];
			final int inCount = in2d.channels();
			for(int ai = startA; ai<endA; ++ai) {
				final ActivationSeq at = as.a[ai];
				if(at==null) continue;
				final ActivationSet a = at.get(t);
				final ActivationSet e = errors.a[ai].get(t);


				for(int bri=startBri; bri<endBri; ++bri) {
					//Copy unmodified inputs
					for(int i=0; i<inCount; ++i) {
						e.add(in, in2d.index(bri, i), e.get(products, product2d.index(bri, i)));
					}
					//Compute quadratic interactions
					for(Kernel krn: kernels) {
						for(int rep=0; rep<krn.repetitions; ++rep) {
							final int leftOffset = krn.leftStart + rep*krn.leftMatDim;
							final int rightOffset = krn.rightStart + rep*krn.rightMatDim;
							final int outOffset = inCount + krn.outStart + rep*krn.outMatDim;
							for(int l = 0, oi=outOffset; l<krn.leftDim; ++l) {
								final int endR = krn.symmetric?l+1:krn.rightDim;
								for(int r = 0; r<endR; ++r, ++oi) {
									final float err = e.get(products, product2d.index(bri, oi))*krn.scaleDown;
									for(int m=0; m<krn.midDim; ++m) {
										final int li = leftOffset + l*krn.midDim + m;
										final int lindex = left2d.index(bri, li);
										final float fl = a.get(leftFactors, lindex);
										final float fr;
										final int rindex;
										if(krn.symmetric) {
											final int ri = leftOffset + r*krn.midDim + m;
											fr = a.get(leftFactors, rindex = left2d.index(bri, ri));
											e.add(leftFactors, rindex, err*fl);
										}else {
											final int ri = rightOffset + r*krn.midDim + m;
											fr = a.get(rightFactors, rindex = right2d.index(bri, ri));
											e.add(rightFactors, rindex, err*fl);
										}
										e.add(leftFactors, lindex, err*fr);
									}
								}
							}
						}
					}

				}
			}
		}else {
			int startA = 0;
			int endA = as.length;
			int inCount = in2d.channels();
			for(int ai = startA; ai<endA; ++ai) {
				ActivationSeq at = as.a[ai];
				if(at==null) continue;
				ActivationSet a = at.get(t);
				ActivationSet e = errors.a[ai].get(t);


				//Copy unmodified inputs
				for(int i=0; i<inCount; ++i) {
					e.add(in, inst, i, e.get(products, inst, i));
				}
				//Compute quadratic interactions
				for(Kernel krn: kernels) {
					for(int rep=0; rep<krn.repetitions; ++rep) {
						int leftOffset = krn.leftStart + rep*krn.leftMatDim;
						int rightOffset = krn.rightStart + rep*krn.rightMatDim;
						int outOffset = inCount + krn.outStart + rep*krn.outMatDim;
						for(int l = 0, oi=outOffset; l<krn.leftDim; ++l) {
							int endR = krn.symmetric?l+1:krn.rightDim;
							for(int r = 0; r<endR; ++r, ++oi) {
								float err = e.get(products, inst, oi)*krn.scaleDown;
								for(int m=0; m<krn.midDim; ++m) {
									int li = leftOffset + l*krn.midDim + m;
									int lindex = left2d.index(inst, li);
									float fl = a.get(leftFactors, lindex);
									float fr;
									int rindex;
									if(krn.symmetric) {
										int ri = leftOffset + r*krn.midDim + m;
										fr = a.get(leftFactors, rindex = left2d.index(inst, ri));
										e.add(leftFactors, rindex, err*fl);
									}else {
										int ri = rightOffset + r*krn.midDim + m;
										fr = a.get(rightFactors, rindex = right2d.index(inst, ri));
										e.add(rightFactors, rindex, err*fl);
									}
									e.add(leftFactors, lindex, err*fr);
								}
							}
						}
					}
				}


			}
		}
		rightModule.backprop(phase, params, as, errors, t, inst);
		leftModule.backprop(phase, params, as, errors, t, inst);
	}


	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, 
			ActivationBatch errors, ParamSet gradients, int t, int[] inst) {
		afterModule.gradients(phase, params, as, errors, gradients, t, inst);
		leftModule.gradients(phase, params, as, errors, gradients, t, inst);
		rightModule.gradients(phase, params, as, errors, gradients, t, inst);
	}

	@Override
	public void regularize(String phase, ParamSet params, ParamSet gradients, float globReg) {
		leftModule.regularize(phase, params, gradients, globReg);
		rightModule.regularize(phase, params, gradients, globReg);
		afterModule.regularize(phase, params, gradients, globReg);
	}
	//	//	@Override
	//	public void declareDependencies(Dependencies d) {
	//		d.declare(new InputDependency(in, this, dt));
	//		d.declare(new OutputDependency(this, out));
	//	}

	@Override
	public void dontComputeInPhase(String phase) {		
	}
	//	@Override
	public boolean wouldBackprop(String phase) {
		return !dontBackprop.contains(phase);
	}
	HashSet<String> dontBackprop=new HashSet<>();

	public QuadraticInteraction dontBackprop(String phase){
		dontBackprop.add(phase);
		return this;
	}
	@Override
	public Interface getIn() {
		return in;
	}
	@Override
	public Interface getOut() {
		return out;
	}

	@Override
	public List<Module> getSubmodules() {
		return Arrays.asList(exec);
	}
	@Override
	public void initParams(ParamSet p) {
		leftModule.initParams(p);
		rightModule.initParams(p);
		afterModule.initParams(p);
	}


	@Override
	public int dt() {
		return 0;
	}


	@Override
	public int[] shift() {
		return null;
	}


} 
