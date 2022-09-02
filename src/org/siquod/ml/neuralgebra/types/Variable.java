package org.siquod.ml.neuralgebra.types;

public class Variable {
	private static int id_counter = 0;
	public final String name;
	public final int id;
	public Variable(String name) {
		this.name = name;
		this.id=0;
	}
	private Variable(String name, int id) {
		this.name = name;
		this.id = id;
	}
	public Variable newId() {
		int id;
		synchronized(Variable.class) {
			id = ++id_counter;
		}
		return new Variable(this.name, id);
	}
}
