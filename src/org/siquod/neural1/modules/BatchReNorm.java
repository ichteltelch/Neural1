package org.siquod.neural1.modules;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ActivationSeq;
import org.siquod.neural1.ActivationSet;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlock;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;

public class BatchReNorm implements InOutScaleBiasModule{

	private static final float epsilon = 1e-10f;
	private Interface in;
	private Interface out;
	private Interface mean, sdev;
	ParamBlock add, mult, runningMean, runningSdev;
	int age=0;
	boolean hasAdd=true;

	Function<Integer, Float> rmaxSchedule=rmaxSchedule(100);
	Function<Integer, Float> dmaxSchedule=dmaxSchedule(100);
	
	static Function<Integer, Float> rmaxSchedule(int scal){
		return n -> n < scal ? 1 : n <8*scal ?  1+2*(n-scal)/7f:3;
	}	
	static Function<Integer, Float> dmaxSchedule(int scal){
		return n -> n < scal ? 0 : n <5*scal ?  5*(n-scal)/4f:5;
	}
	public BatchReNorm(boolean hasAdd) {
		this.hasAdd=hasAdd;
	}
	public BatchReNorm() {
		this(true);
	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		in = ia.get("in");
		out = ia.get("out");
		if(in.count!=out.count)
			throw new IllegalArgumentException("input and output layer must be of the same size");

	}

	@Override
	public void allocateStatistics(InterfaceAllocator ia) {
		ia.push(null);
		mean = ia.allocate(new Interface("mean", in.tf));
		sdev = ia.allocate(new Interface("sdev", in.tf));
		ia.pop();
	}
	@Override
	public void allocate(ParamAllocator ia) {
		if(hasAdd) add = ia.allocate(new ParamBlock("add", in.count));
		mult = ia.allocate(new ParamBlock("mult", in.count));
		runningMean = ia.allocate(new ParamBlock("mean", in.count));
		runningSdev = ia.allocate(new ParamBlock("sdev", in.count));
	}

	@Override
	public void share(ParamBlocks ps) {
		if(hasAdd) add = ps.get("add", in.count);
		mult = ps.get("mult", in.count);		
		runningMean = ps.get("mean", in.count);
		runningSdev = ps.get("sdev", in.count);		
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret = new ParamBlocks("Conv1D");
		if(hasAdd) ret.add(add);
		ret.add(mult);
		ret.add(runningMean);
		ret.add(runningSdev);
		return ret;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		ActivationSet bparm = as.batchParams.get(t);
		if(inst==null) {
			if(training != ForwardPhase.TESTING) {
				float dmax=dmaxSchedule.apply(age);
				float rmax=rmaxSchedule.apply(age);
				for(int i=0; i<in.count; ++i) {
					float mean=0;
					int count=0;
					for(ActivationSeq b: as.a) {
						if(b==null)
							continue;
						mean+=b.get(t).get(in, i);
						count++;
					}
					mean/=count;
					float var = 0;
					for(ActivationSeq b: as.a) {
						if(b==null)
							continue;
						float d = b.get(t).get(in, i) - mean;
						var += d*d;
					}
					var/=count;
					float sdev=(float) Math.sqrt(var+epsilon);
					float scal = 1/sdev;
					float rmean=params.get(runningMean, i);
					float rsdev=params.get(runningSdev, i);

					float r=Math.min(rmax, Math.max(1/rmax, sdev/rsdev));
					float d=Math.min(dmax, Math.max(-dmax, (mean-rmean)/rsdev));
					{
						bparm.set(this.mean, i, mean);
						bparm.set(this.sdev, i, sdev);
					}

					float addVal=hasAdd?params.get(add, i):0;
					float multVal=params.get(mult, i);
//					if(i==0) {
//						System.out.println("mean = "+mean+", var = "+var);
//						System.out.println("add = "+addVal+", mult = "+multVal);
//					}
					//					System.out.println(" (x - "+mean+") * "+scal+" * "+multVal+" + "+addVal);
					for(ActivationSeq b: as.a) {
						if(b==null)
							continue;
						ActivationSet a = b.get(t);
						float dv = (a.get(in, i) - mean)*scal*r+d;
						float v = dv*multVal + addVal;
						a.add(out, i, v);
					}
				}
			}else {
				for(ActivationSeq b: as.a) {
					if(b==null)
						continue;
					for(int i=0; i<in.count; ++i) {
						ActivationSet a = b.get(t);
						float mean = params.get(this.runningMean, i);
						float sdev = params.get(this.runningSdev, i);
						float addVal=hasAdd?params.get(add, i):0;
						float multVal=params.get(mult, i);
//						if(i==0) {
//							System.out.println("mean = "+mean+", var = "+var);
//							System.out.println("add = "+addVal+", mult = "+multVal);
//						}
						//						System.out.println(" (x + "+mShift+") * "+vScale+" * "+multVal+" + "+addVal);

						float scal = 1/sdev;

						float d = (a.get(in, i) - mean)*scal;
						float v = d*multVal + addVal;
						a.add(out, i, v);
					}
				}
			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}

	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		float dmax=dmaxSchedule.apply(age);
		float rmax=rmaxSchedule.apply(age);
		if(inst==null) {
			for(int i=0; i<in.count; ++i) {

				float mean = as.batchParams.get(t).get(this.mean, i);
				float sdev = as.batchParams.get(t).get(this.sdev, i);
				float scal = 1/sdev;
				float rmean=params.get(runningMean, i);
				float rsdev=params.get(runningSdev, i);

				float r=Math.min(rmax, Math.max(1/rmax, sdev/rsdev));
				float d=Math.min(dmax, Math.max(-dmax, (mean-rmean)/rsdev));

				//				float addVal=params.get(add, i);
				float multVal=params.get(mult, i);
				int count=0;
				float scalErr=0;

				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					count++;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float me=e.get(out, i)*multVal;
					float scalInp = (a.get(in, i)-mean);
					scalErr+=me*scalInp;


				}
				//			float scal = 1/Math.sqrt(var+epsilon);
				//			ð var = -0.5*(var+epsilon)^1.5 * ð scal;
				float varErr = scalErr * -0.5f / (sdev*sdev*sdev);
				varErr/=count;
				float meanErr=0;
				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float scalInp = a.get(in, i)-mean;
					float me=e.get(out, i)*multVal*r;
					me *= scal;
					me += 2 * scalInp * varErr;
					meanErr += -me;
				}
				meanErr/=count;
				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float scalInp = a.get(in, i)-mean;
					float me=e.get(out, i)*multVal*r;
					me *= scal;
					me += 2 * scalInp * varErr;
					me += meanErr;
					e.add(in, i, me);
				}
			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}
	}

	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		float dmax=dmaxSchedule.apply(age);
		float rmax=rmaxSchedule.apply(age);
		if(inst==null) {
			for(int i=0; i<in.count; ++i) {

				float mean = as.batchParams.get(t).get(this.mean, i);
				float sdev = as.batchParams.get(t).get(this.sdev, i);
				float scal = 1/sdev;
				float rmean=params.get(runningMean, i);
				float rsdev=params.get(runningSdev, i);

				float r=Math.min(rmax, Math.max(1/rmax, sdev/rsdev));
				float d=Math.min(dmax, Math.max(-dmax, (mean-rmean)/rsdev));

				//				int count=0;

				float addErr=0;
				float multErr=0;
				for(int b=0; b<as.length; ++b) {
					ActivationSeq es=errors.a[b];
					if(es==null)
						continue;
					//					count++;
					ActivationSet a = as.a[b].get(t);
					ActivationSet e = es.get(t);
					float oe=e.get(out, i);
					addErr += oe;
					float reparin = (a.get(in, i)-mean)*scal*r+d;
					multErr += reparin * oe;


				}
				if(hasAdd)
					gradients.add(add, i, addErr);
				gradients.add(mult, i, multErr);
//				gradients.set(runningMean, i, 0);
//				gradients.add(runningVar, i, 0);
			}
		}else {
			//TODO
			throw new UnsupportedOperationException("Not implemented");
		}	
	}
	@Override
	public void dontComputeInPhase(String phase) {

	}
	@Override
	public void initParams(ParamSet p) {
		age=0;
		for(int i=0; i<in.count; ++i) {
			if(hasAdd)
				p.set(add, i, 0);
			p.set(mult, i, 1);
			p.set(runningMean, i, 0);
			p.set(runningSdev, i, 1);
		}
	}

	@Override
	public List<Module> getSubmodules() {
		return Collections.emptyList();
	}

	@Override
	public Interface getIn() {
		return in;
	}

	@Override
	public Interface getOut() {
		return out;
	}

	@Override
	public int dt() {
		return 0;
	}

	@Override
	public int[] shift() {
		return null;
	}
	@Override
	public void updateStatistics(ActivationSeq stat, ParamSet params, Function<Integer, Float> owt_fun, float[] weight, int tMin) {
		float owt=owt_fun.apply(age++);
		System.out.print("");
		for(int i=0; i<in.count; ++i) {
			float smean, ssdev, swt;
			smean=params.get(runningMean, i)*owt;
			ssdev=params.get(runningSdev, i)*owt;
			swt=owt;
			for(int ts=0; ts<weight.length; ++ts) {
				int t = ts + tMin;
				float df=weight[ts];
				swt+=df;
				smean += df*stat.get(t).get(mean, i);
				ssdev += df*stat.get(t).get(sdev, i);
			}
			float mv = smean/swt;
			float vv = ssdev/swt;
			params.set(runningMean, i, mv);
			params.set(runningSdev, i, vv);
		}
	}
	public ParamBlock getAdd() {
		return add;
	}
	public ParamBlock getMult() {
		return mult;
	}
	public ParamBlock getRunningMean() {
		return runningMean;
	}
	public ParamBlock getRunningVar() {
		return runningSdev;
	}
	@Override
	public ParamBlock getBias() {
		return add;
	}
	@Override
	public ParamBlock getScale() {
		return mult;
	}

}
