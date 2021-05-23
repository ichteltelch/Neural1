package org.siquod.neural1;
//package neural;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collection;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Set;
//
//import neural.DependencyGraph.MNode;
//
//public class DependencyGraph {
//	Dependencies dep;
//	public enum Type{
//		FEED_FORWARD,
//		BIDIRECTIONAL,
//		FORWARD_RECURRENT,
//		BACKWARD_RECURRENT
//	}
//	Type type;
//	public class MNode{
//		public final Module m;
//		public final int t;
//		public final HashSet<INode> dependOn=new HashSet<>();
//		boolean processing;
//		public HashSet<INode> successors=new HashSet<>();
//		public MNode(Module m2, int t2) {
//			m=m2;
//			t=t2;
//		}
//		@Override
//		public int hashCode() {
//			return m.hashCode()+t*1237;
//		}
//		@Override
//		public boolean equals(Object o) {
//			if (o == this)
//				return true;
//			if (o == null)
//				return false;
//			if (o instanceof DependencyGraph.MNode) {
//				DependencyGraph.MNode a = (DependencyGraph.MNode) o;
//				if(t!=a.t) return false;
//				if(!m.equals(a.m)) return false;
//				return true;
//			}
//			return false;
//		}
//		@Override
//		public String toString() {
//			return m.toString()+" fed from "+dependOn;
//		}
//	}
//	public class INode{
//		public final Interface i;
//		public final int t;
//		HashSet<MNode> dependOn=new HashSet<>();
//		public INode(Interface i2, int t2) {
//			i=i2;
//			t=t2;
//		}
//		@Override
//		public int hashCode() {
//			return i.hashCode()+t*1237;
//		}
//		@Override
//		public boolean equals(Object o) {
//			if (o == this)
//				return true;
//			if (o == null)
//				return false;
//			if (o instanceof DependencyGraph.INode) {
//				DependencyGraph.INode a = (DependencyGraph.INode) o;
//				if(t!=a.t) return false;
//				if(!i.equals(a.i)) return false;
//				return true;
//			}
//			return false;
//		}
//		@Override
//		public String toString() {
//			return i.name +" | at time "+t;
//		}
//	}
//	public DependencyGraph(Dependencies dep){
//		this.dep=dep;
//		type = dep.needsFuture()?
//				dep.needsPast()?Type.BIDIRECTIONAL:Type.BACKWARD_RECURRENT:
//					dep.needsPast()?Type.FORWARD_RECURRENT:Type.FEED_FORWARD;
//	}
//	public static <E> E canonical(HashMap<E, E> canon, E n){
//		E ret = canon.get(n);
//		if(ret==null){
//			canon.put(n, n);
//			return n;
//		}
//		return ret;
//	}
//	public ArrayList<MNode> makePlan(Collection<String> phases, int duration, boolean skipNops){
//		ArrayList<MNode> plan=new ArrayList<>();
//		HashMap<MNode, MNode> mCanon=new HashMap<>();
//		HashMap<INode, INode> iCanon=new HashMap<>();
//		switch(type){
//		case FEED_FORWARD:
//		case FORWARD_RECURRENT:
//		case BACKWARD_RECURRENT:
//			make_Plan(plan, phases, mCanon, iCanon, 0, 0, skipNops);
//			break;
//		case BIDIRECTIONAL:
//			duration--;
//			make_Plan(plan, phases, mCanon, iCanon, -duration/2, duration-duration/2, skipNops);
//		}
//		return plan;
//	}
//	private void make_Plan(ArrayList<MNode> plan, Collection<String> phases, HashMap<MNode, MNode> mCanon, HashMap<INode, INode> iCanon, int tMin, int tMax, boolean skipNops) {
//		for(Interface i: dep.ifIn.keySet()){
//			boolean shouldCompute=false;
//			for(String phase: phases){
//				if(i.shouldCompute(phase)){
//					shouldCompute=true;
//					break;
//				}
//			}
//			if(!shouldCompute)
//				continue;
//			INode nn = new INode(i, 0);
//			INode in=canonical(iCanon, nn);
//			if(nn!=in)
//				continue;
//			fillPlan(in, plan, mCanon, iCanon, tMin, tMax, skipNops);
//
//		}
//
//
//	}
//	private void fillPlan(INode in, ArrayList<MNode> plan, HashMap<MNode, MNode> mCanon, HashMap<INode, INode> iCanon, int tMin, int tMax, boolean skipNops) {
//		ArrayList<OutputDependency> prev = dep.ifIn.get(in.i);
//		if(prev==null)
//			return;
//		for(OutputDependency m: prev){
//			MNode nn=new MNode(m.via, 0);
//			MNode mn=canonical(mCanon, nn);
//			in.dependOn.add(mn);
//			mn.successors.add(in);
//
//			if(nn!=mn)
//				continue;
//			fillPlan(mn, plan, mCanon, iCanon, tMin, tMax, skipNops);
//		}
//
//	}
//	private void fillPlan(MNode mn, ArrayList<MNode> plan, HashMap<MNode, MNode> mCanon, HashMap<INode, INode> iCanon, int tMin, int tMax, boolean skipNops) {
//		ArrayList<InputDependency> prev = dep.modIn.get(mn.m);
//		if(mn.processing)
//			throw new IllegalStateException("The network is not acyclic");
//		try{
//			mn.processing=true;
//			for(InputDependency m: prev){
//				int t = mn.t + m.dt;
//				if(t<tMin || t>tMax)
//					continue;
//				INode nn = new INode(m.in, t);
//				INode in=canonical(iCanon, nn);
//				mn.dependOn.add(in);
//				if(nn!=in)
//					continue;
//				fillPlan(in, plan, mCanon, iCanon, tMin, tMax, skipNops);
//			}
////			if(!(mn.m instanceof NopCastLayer))
//				plan.add(mn);
//		}finally{
//			mn.processing=false;
//		}
//	}
//	public ArrayList<MNode> makePlan(int duration, boolean skipNops, String... phases) {
//		return makePlan(Arrays.asList(phases), duration, skipNops);
//	}
//
//}
