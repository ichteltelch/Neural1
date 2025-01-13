package org.siquod.ml.data;

public interface CursorTransformer {
	default int dimAfterTransform(int dimBeforeTransform) {
		return dimBeforeTransform;
	}
	default void transform(double[] input) {
		
	}
}
