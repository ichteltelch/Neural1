package org.siquod.ml.neural1.modules;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.Consumer;

import org.siquod.ml.neural1.modules.QuadraticInteraction.Kernel;
import org.siquod.ml.neural1.modules.regularizer.L2Reg;
import org.siquod.ml.neural1.modules.regularizer.Regularizer;

public class QuadraticInteractionBuilder{
	public static final Regularizer DEFAULT_REG =new L2Reg(.00001); 
	@FunctionalInterface
	public static interface Configurator extends Consumer<QuadraticInteractionBuilder>{}
	public static final Configurator PRINT = q->System.out.println(q.kernels);
	public static final InOutFactory PRODUCE_DEFAULT = (i, o) ->	new Dense(true).regularizer(DEFAULT_REG);
	public static final InOutFactory PRODUCE_DEFAULT_NOBIAS = (i, o) ->	new Dense(false).regularizer(DEFAULT_REG);
	int leftAt = 0;
	int rightAt = 0;
	int outAt = 0;
	int repetitions = 1;
	float scaleDown = .1f;
	InOutFactory produceLeft = PRODUCE_DEFAULT;
	InOutFactory produceRight = PRODUCE_DEFAULT;
	InOutFactory produceProduct = PRODUCE_DEFAULT;
	ArrayList<Kernel> kernels = new ArrayList<>();
	public QuadraticInteractionBuilder config(Consumer<? super QuadraticInteractionBuilder> configurator) {
		configurator.accept(this);
		return this;
	}
	public QuadraticInteractionBuilder kernel(int left, int mid, int right) {
		Kernel nk = new Kernel(left, mid, right, repetitions, leftAt, rightAt, outAt, scaleDown);
		leftAt = nk.leftEnd;
		rightAt = nk.rightEnd;
		outAt = nk.outEnd;
		kernels.add(nk);
		return this;
	}
	public QuadraticInteractionBuilder symmetricKernel(int outer, int inner) {
		Kernel nk = new Kernel(outer, inner, repetitions, leftAt, rightAt, outAt, scaleDown);
		leftAt = nk.leftEnd;
		rightAt = nk.rightEnd;
		outAt = nk.outEnd;
		kernels.add(nk);
		return this;
	}
	public QuadraticInteractionBuilder repetitions(int rep) {
		repetitions = rep;
		return this;
	}

	public QuadraticInteractionBuilder leftModule(InOutFactory f) {
		Objects.requireNonNull(f);
		produceLeft = f;
		return this;
	}
	public QuadraticInteractionBuilder rightModule(InOutFactory f) {
		Objects.requireNonNull(f);
		produceRight = f;
		return this;
	}
	public QuadraticInteractionBuilder bothModules(InOutFactory f) {
		Objects.requireNonNull(f);
		produceLeft = f;
		produceRight = f;
		return this;
	}
	public QuadraticInteractionBuilder scale(double s) {
		scaleDown = (float) s;
		return this;
	}
	public QuadraticInteractionBuilder outputModule(InOutFactory f) {
		Objects.requireNonNull(f);
		produceProduct = f;
		return this;
	}
	public Kernel[] getKernels() {
		return kernels.toArray(new Kernel[kernels.size()]);
	}
	public QuadraticInteraction build() {
		if(kernels.isEmpty())
			throw new IllegalArgumentException("must specify at least one kernel");
		return new QuadraticInteraction(
				getKernels(), 
				produceLeft, 
				produceRight,
				produceProduct,
				leftAt, rightAt, outAt);
	}
}