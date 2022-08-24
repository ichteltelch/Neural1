package org.siquod.neuralgebra.types;

import org.siquod.neuralgebra.constraints.Constraints;
import org.siquod.neuralgebra.constraints.UnificationFailure;

public class PrimType extends AType{
	public static final PrimType BOOL = new PrimType("bool");
	public static final PrimType I32 = new PrimType("i32");
	public static final PrimType U31 = new PrimType("u31");
	public static final PrimType F32 = new PrimType("f32");

	
	
	public final String name;
	public PrimType (String name) {
		super(new Constraints());
		this.name = name;	
	}
	@Override
	public AType unify(AType o) {
		if(o==this)
			return this;
		return new Void(Constraints.b().fail(new UnificationFailure(this, o)).build());
	}
	
	
}
