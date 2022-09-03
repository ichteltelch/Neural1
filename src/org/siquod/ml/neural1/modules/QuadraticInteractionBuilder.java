package org.siquod.ml.neural1.modules;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.Consumer;

import org.siquod.ml.neural1.modules.QuadraticInteraction.Kernel;

public class QuadraticInteractionBuilder{
	@FunctionalInterface
	public static interface Configurator extends Consumer<QuadraticInteractionBuilder>{}
	public static final Configurator PRINT = q->System.out.println(q.kernels);
	public static final InOutFactory PRODUCE_DEFAULT = (i, o) ->	new Dense(true);
	public static final InOutFactory PRODUCE_DEFAULT_NOBIAS = (i, o) ->	new Dense(false);
	int leftAt = 0;
	int rightAt = 0;
	int outAt = 0;
	int repetitions = 1;
	InOutFactory produceLeft = PRODUCE_DEFAULT;
	InOutFactory produceRight = PRODUCE_DEFAULT;
	InOutFactory produceProduct = PRODUCE_DEFAULT_NOBIAS;
	ArrayList<Kernel> kernels = new ArrayList<>();
	public QuadraticInteractionBuilder config(Consumer<? super QuadraticInteractionBuilder> configurator) {
		configurator.accept(this);
		return this;
	}
	public QuadraticInteractionBuilder kernel(int left, int mid, int right) {
		Kernel nk = new Kernel(left, mid, right, repetitions, leftAt, rightAt, outAt);
		leftAt = nk.leftEnd;
		rightAt = nk.rightEnd;
		outAt = nk.outEnd;
		kernels.add(nk);
		return this;
	}
	public QuadraticInteractionBuilder symmetricKernel(int outer, int inner) {
		Kernel nk = new Kernel(outer, inner, repetitions, leftAt, rightAt, outAt);
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