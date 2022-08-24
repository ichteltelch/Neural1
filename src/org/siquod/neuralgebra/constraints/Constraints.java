package org.siquod.neuralgebra.constraints;

import java.util.ArrayList;

public class Constraints {
	ArrayList<UnificationFailure> fail=new ArrayList<>();
	ArrayList<EqConstraint> eq=new ArrayList<>();
	public Constraints unify(Constraints o) {
		return new Constraints();
	}
	public static Builder b() {return new Builder();}
	public static class Builder{
		Constraints bb = new Constraints();
		public Builder unify(Constraints c) {
			for(UnificationFailure f: c.fail)
				fail(f);
			for(EqConstraint e: c.eq)
				add(e);
			return this;
		}
		public Builder fail(UnificationFailure f) {
			bb.fail.add(f);
			return this;
		}
		public Builder add(EqConstraint e) {
			bb.eq.add(e);
			return this;
		}
		public Constraints build() {return bb;}
	}
}
