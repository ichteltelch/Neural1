package org.siquod.neural1;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class InterfaceAllocator {
	int counter=0;
	int dc=0;
	int doc=0;
//	int lifeCounter=0;

	Stack<HashMap<String, Interface>> bindings=new Stack<>();
	{
		bindings.push(new HashMap<String, Interface>());
	}
	public int getCount(){
		return counter;
	}
	public int getDecisionCount(){
		return dc;
	}
	public int getDropoutCount() {
		return doc;
	}
	public int allocateDecision(int amount) {
		int ret=dc;
		dc+=amount;
		return ret;
	}
	public int allocateDropout(int amount){
		int ret=doc;
		doc+=amount;
		return ret;
	}
	public Interface allocate(Interface i){
		HashMap<String, Interface> top = bindings.peek();
		Interface old = top.get(i.name);
		if(old==null){
			if(i.offset==-1) {
				i.offset=counter;
				counter += i.count;
			}
			top.put(i.name, i);
			return i;
		}else{
			if(i.count!=old.count){
				throw new IllegalArgumentException("There is already an interface called '"+i.name+"', and it has different width.");
			}
			i.offset=old.offset;

			return old;
		}
	}
	public void put(Interface i) {
		HashMap<String, Interface> top = bindings.peek();
		Interface old = top.get(i.name);
		if(i.count!=old.count){
			throw new IllegalArgumentException("There is already an interface called '"+i.name+"'.");
		}
		top.put(i.name, i);
	}
	public Interface get(String name){
		HashMap<String, Interface> top = bindings.peek();
		Interface ret = top.get(name);
		if(ret==null)
			throw new IllegalArgumentException("Interface '"+name+"' is not in scope");
		return ret;
	}
	public Interface get(String name, int expectedSize){
		HashMap<String, Interface> top = bindings.peek();
		Interface ret = top.get(name);
		if(ret==null)
			throw new IllegalArgumentException("Interface '"+name+"' is not in scope");
		if(ret.count!=expectedSize)
			throw new IllegalArgumentException("Interface '"+name+"' does not have the expected size");
		return ret;
	}
	public void push(Map<String, String> params){
		HashMap<String, Interface> newTop=new HashMap<String, Interface>();
		if(params!=null && !params.isEmpty()){
			HashMap<String, Interface> oldTop=bindings.peek();
			for(String parname: params.keySet()){
				String s = params.get(parname);
				Interface interf = oldTop.get(s);
				if(interf==null)
					throw new IllegalArgumentException("Outer environment has no interface named '"+s+"'");
				if(newTop.containsKey(parname))
					throw new IllegalArgumentException("Duplicate parameter name '"+parname+"'");
				newTop.put(parname, interf);
			}
		}
		bindings.push(newTop);
	}
	public void pop(){
		bindings.pop();
	}
	public ActivationSeq makeSeq(int length) {
		return new ActivationSeq(this, length, doc);
	}
	public ActivationSet makeSet() {
		return new ActivationSet(counter, dc);
	}
//	public int allocateLifeIndex() {
//		return lifeCounter++;
//	}
	
}
