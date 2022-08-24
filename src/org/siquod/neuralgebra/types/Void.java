package org.siquod.neuralgebra.types;

import org.siquod.neuralgebra.constraints.Constraints;

public class Void extends AType{
	public Void(Constraints c) {
		super(c);
	}
	public Void(Constraints c1, Constraints c2) {
		super(c1.unify(c2));
	}

	@Override
	public AType unify(AType o) {
		return new Void(constraints, o.constraints);
	}
}
