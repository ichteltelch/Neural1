package org.siquod.ml.neuralgebra.types;

import org.siquod.ml.neuralgebra.constraints.Constraints;

public abstract class AType {
	public final Constraints constraints;
	public AType(Constraints c) {
		constraints = c;
	}
	public abstract AType unify(AType o) ;
}
