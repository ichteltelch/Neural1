package org.siquod.neural1;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class ParamAllocator {
	int counter=0;
	Stack<HashMap<String, ParamBlock>> bindings=new Stack<>();
	{
		bindings.push(new HashMap<String, ParamBlock>());
	}
	public int getCount(){
		return counter;
	}
	public ParamBlock allocate(ParamBlock i){
		HashMap<String, ParamBlock> top = bindings.peek();
		ParamBlock old = top.get(i.name);
		if(old==null){
			i.start=counter;
			counter += i.count;
			top.put(i.name, i);
			return i;
		}else if(old instanceof ParamBlock){
			ParamBlock old_ = (ParamBlock) old;
			if(i.count!=old_.count){
				throw new IllegalArgumentException("There is already a param block called '"+i.name+"', and it has different size.");
			}
			i.start=old_.start;
			return old_;
		}else{
			throw new IllegalArgumentException("There is already a param block called '"+i.name+"', but it is compound.");
		}
	}
	public ParamBlock get(String name){
		HashMap<String, ParamBlock> top = bindings.peek();
		ParamBlock ret = top.get(name);
		if(ret==null)
			throw new IllegalArgumentException("Param block '"+name+"' is not in scope");
		return ret;
	}
	public void push(Map<String, String> params){
		HashMap<String, ParamBlock> newTop=new HashMap<String, ParamBlock>();
		if(params!=null && !params.isEmpty()){
			HashMap<String, ParamBlock> oldTop=bindings.peek();
			for(String s: params.keySet()){
				ParamBlock interf = oldTop.get(s);
				if(interf==null)
					throw new IllegalArgumentException("Outer environment has no param block named '"+s+"'");
				String parname = params.get(s);
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
}
