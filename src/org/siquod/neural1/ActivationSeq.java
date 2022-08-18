package org.siquod.neural1;

import java.util.Random;

public class ActivationSeq implements Cloneable{
	ActivationSet[] steps;
	int[] dropoutMasks;
//	int[] lifeInterfaces;
	int start;
	int begin, end;
	InterfaceAllocator ia;
	public ActivationSeq(InterfaceAllocator ia, int length, int doCount){
		steps=new ActivationSet[length];
		dropoutMasks=new int[doCount];
//		lifeInterfaces=new int[lifeCount];
		this.ia=ia;
	}
	public void setBuffer(int i, ActivationSet a){
		steps[i]=a;
	}
	public void clear(){
		for(ActivationSet a: steps)
			a.clear();
		start=end=begin=0;
	}
//	public void clearLifeInterfaces(){
//		Arrays.fill(lifeInterfaces, 0);
//	}
//	public void clearLifeInterfaces(boolean life){
//		Arrays.fill(lifeInterfaces, life?1:0);
//	}

	public ActivationSeq init(){
		for(int i=0; i<steps.length; ++i)
			steps[i]=ia.makeSet();
		return this;
	}
	public ActivationSeq entire(){
		start=begin=0;
		end=steps.length;
		return this;
	}
	public void advance(){
		end ++;
		if(end-begin>steps.length){
			begin++;
			start++;
			start%=steps.length;
		}
		
		ActivationSet as = get(end-1);
		as.clear();
	}
	public ActivationSet get(int t){
		if(t<begin)
			return null;
		if(t>=end)
			return null;
		return steps[((t-begin)+start)%steps.length];
	}
	public void sampleDropout(int dropoutOffset, int dropoutCount, double keepProb, Random rand){
		for(int j=0; j<dropoutCount; ++j){
			dropoutMasks[dropoutOffset+j] = rand.nextDouble()<keepProb?1:0;
//			dropoutMasks[dropoutOffset+j] = Math.random()<keepProb?1:0;
		}
	}
	public void sampleDropout(int dropoutOffset, int dropoutCount, double keepProb){
		for(int j=0; j<dropoutCount; ++j){
			dropoutMasks[dropoutOffset+j] = Math.random()<keepProb?1:0;
		}
	}
	public int getDropout(int index){
		return dropoutMasks[index];
	}
	public void setDropout(int index, int value){
		dropoutMasks[index]=value;
	}
	@Override
	public ActivationSeq clone() {
		try {
			ActivationSeq r = (ActivationSeq) super.clone();
			r.dropoutMasks=dropoutMasks.clone();
			r.steps=steps.clone();
			for(int i=0; i<steps.length; ++i)
				r.steps[i]=steps[i].clone();

			return r;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			return null;
		}
	}
//	public boolean isDead(Interface in) {
//		if(in.lifeIndex==-1)return false;
//		return lifeInterfaces[in.lifeIndex]!=1;
//	}
//	public void kill(Interface in){
//		if(in.lifeIndex==-1)return;
//		lifeInterfaces[in.lifeIndex]=-1;
//	}
//	public void propagateLife(Interface in){
//		if(in.lifeIndex==-1)return;
//		if(lifeInterfaces[in.lifeIndex]<1)
//		lifeInterfaces[in.lifeIndex]++;
//	}
}
