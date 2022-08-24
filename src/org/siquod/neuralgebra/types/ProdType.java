package org.siquod.neuralgebra.types;

import java.util.ArrayList;

import org.siquod.neuralgebra.constraints.Constraints;

public class ProdType extends AType{
	final AType[] factors;
	final String[] names;
	final int pc;
	
	ProdType(String[] names, AType[] factors, Constraints cs) {
		super(cs);
		this.factors = factors;
		this.names = names;
		int pc = 0;
		for(; pc<names.length && names[pc]==null; ++pc);
		this.pc = pc;
	}
	public static Builder b() {return new Builder();}
	public static class Builder{
		ArrayList<String> names=new ArrayList<>();
		ArrayList<AType> types=new ArrayList<>();
		Constraints.Builder cb = Constraints.b();
		public Builder factor(String name, AType type) {
			if(name==null)
				return factor(type);
			names.add(name);
			types.add(type);
			cb.unify(type.constraints);

			return this;
		}
		public Builder factor(AType type) {
			if(names.isEmpty() || names.get(names.size())==null) {
				names.add(null);
				types.add(type);
				cb.unify(type.constraints);
			}else {
				throw new IllegalStateException();
			}
			return this;
		}
		public Builder factors(AType... types) {
			if(names.isEmpty() || names.get(names.size())==null) {
				for(AType t: types) {
					names.add(null);
					this.types.add(t);
					cb.unify(t.constraints);
				}
			}else {
				throw new IllegalStateException();
			}
			return this;
		}
		public ProdType build() {
			return new ProdType(
					names.toArray(new String[names.size()]), 
					types.toArray(new AType[types.size()]),
					cb.build());
		}
	}
	@Override
	public AType unify(AType o) {
		if(!(o instanceof ProdType))
			return new Void(constraints, o.constraints);
		ProdType ot = (ProdType) o;
		if(factors.length != ot.factors.length)
			return new Void(constraints, o.constraints);
		Builder b = b();
		int i=0;
		for(; i<factors.length; ++i) {
			String n1 = names[i];
			String n2 = ot.names[i];
			String n;
			if(n1==null)
				n=n2;
			else if(n2==null)
				n=n1;
			else if(n2.equals(n1))
				n=n1;
			else
				return new Void(constraints, o.constraints);
			b.factor(n, factors[i].unify(ot.factors[i]));			
		}
		return b.build();
	}
}
