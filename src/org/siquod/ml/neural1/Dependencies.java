package org.siquod.ml.neural1;
//package neural;
//
//import java.util.ArrayList;
//import java.util.HashMap;
//
//public class Dependencies {
//	HashMap<Module, ArrayList<InputDependency>> modIn=new HashMap<>();
//	HashMap<Module, ArrayList<OutputDependency>> modOut=new HashMap<>();
//	HashMap<Interface, ArrayList<OutputDependency>> ifIn=new HashMap<>();
//	HashMap<Interface, ArrayList<InputDependency>> ifOut=new HashMap<>();
//	HashMap<String, ArrayList<Interface>> losses=new HashMap<>();
//	boolean needsFuture, needsPast;
//	static <K, V> void mmPut(HashMap<K, ArrayList<V>> m, K k, V v){
//		ArrayList<V> l = m.get(k);
//		if(l==null){
//			m.put(k, l=new ArrayList<>(3));
//		}
//		l.add(v);
//	}
//	public void declare(InputDependency d){
//		mmPut(modIn, d.via, d);
//		mmPut(ifOut, d.in, d);
//		if(d.dt<0)
//			needsPast=true;
//		if(d.dt>0)
//			needsFuture=true;
//	}
//	public void declare(OutputDependency d){
//		mmPut(modOut, d.via, d);
//		mmPut(ifIn, d.out, d);
//	}
//	public void declareLoss(Interface l, String phase){
//		ArrayList<Interface> list=losses.get("phase");
//		if(list==null)
//			losses.put(phase, list=new ArrayList<>());
//		list.add(l);
//	}
//	public boolean isRecurrent(){
//		return needsFuture || needsPast;
//	}
//	public boolean isBidirectional(){
//		return needsFuture && needsPast;
//	}
//	public boolean needsFuture(){
//		return needsFuture;
//	}
//	public boolean needsPast(){
//		return needsPast;
//	}
//	public DependencyGraph createGraph(InterfaceAllocator ia) {
//		for(Interface i: ifOut.keySet())
//			i.lifeIndex=ia.allocateLifeIndex();
//		return new DependencyGraph(this);
//	}
//
//
//}
