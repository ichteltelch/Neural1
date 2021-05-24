package org.siquod.neural1.modules;

import java.util.ArrayList;
import java.util.List;

import org.siquod.neural1.ActivationBatch;
import org.siquod.neural1.ForwardPhase;
import org.siquod.neural1.Interface;
import org.siquod.neural1.InterfaceAllocator;
import org.siquod.neural1.Module;
import org.siquod.neural1.ParamAllocator;
import org.siquod.neural1.ParamBlocks;
import org.siquod.neural1.ParamSet;
import org.siquod.neural1.TensorFormat;

/**
 * 
 * A stack of modules, each having an input and an output.
 * @author bb
 *
 */
public class StackModule implements InOutModule{
	ArrayList<Module> exec=new ArrayList<>();
	ArrayList<InOutModule> layers=new ArrayList<>();
	ArrayList<Interface> shortcutIn=new ArrayList<>();
	ArrayList<Interface> shortcutOut=new ArrayList<>();
	ArrayList<Integer> shortcutWidth=new ArrayList<>();
	ArrayList<Copy> shortcuts=new ArrayList<>();
	Interface in, out;
	ArrayList<Interface> hidden=new ArrayList<>();
	public StackModule(){
	}
	
	public StackModule addFinalLayer(InOutModule m){
		return addLayer(m, (Interface)null);
	}
	public InOutModule addFinalLayer(int outSize, InOutModule... ms) {
		return addFinalLayer(new TensorFormat(outSize), ms);
	}
	public InOutModule addFinalLayer(TensorFormat outSize, InOutModule... ms) {
		return addLayer(outSize, true, ms);
	}
	public StackModule shortcut(){
		return shortcut(-1);
	}
	public StackModule shortcut(int width){
		if(shortcutIn.size()!=shortcutOut.size())
			endShortcut();
		shortcutWidth.add(width);
		shortcutIn.add(hidden.isEmpty()?null:hidden.get(hidden.size()-1));
		return this;
	}
	public StackModule endShortcut() {
		if(shortcutIn.size()==shortcutOut.size())
			return this;
		shortcutOut.add(hidden.isEmpty()?null:hidden.get(hidden.size()-1));
		return this;
	}

	public StackModule addLayer(InOutCastFactory... fs){
		if(layers.size()!=hidden.size())
			throw new IllegalStateException("The final layer has already been added");
		if(hidden.isEmpty())
			throw new IllegalStateException("The first layer cannot be made by a factory");
		Interface last = hidden.get(hidden.size()-1);
		for(InOutCastFactory f: fs){
			InOutCastLayer l =f.produce(last);
			addLayer(l, last = l.getOutput());
		}
		return this;
	}
	public StackModule addLayer(int outSize, InOutModule... ms){
		return addLayer(new TensorFormat(outSize), ms);
	}
	public StackModule addLayer(TensorFormat outSize, InOutModule... ms){
		return addLayer(outSize, false, ms);
	}
	public StackModule addLayer(TensorFormat outSize, boolean last, InOutModule... ms){
		if(last){
			for(int i=0; i<ms.length-1; ++i)
				addLayer(ms[i], outSize);
			addFinalLayer(ms[ms.length-1]);
		}else{
			for(InOutModule m: ms)
				addLayer(m, outSize);
		}
		return this;
	}
	public StackModule addLayer(InOutModule m, TensorFormat outSize){
		return addLayer(m, new Interface(outSize));
	}
	public StackModule addLayer(InOutModule m, Interface out){
		if(layers.size()!=hidden.size())
			throw new IllegalStateException("The final layer has already been added");
		if(out!=null){
			if(out.name!=null)
				out.name += " #"+hidden.size();
			else
				out.name = "hidden Layer #"+hidden.size();
			hidden.add(out);
		}
		layers.add(m);
		return this;
	}

	@Override
	public void allocate(InterfaceAllocator ia) {
		if(layers.size()==hidden.size()){
			layers.add(new Copy(0, new int[hidden.get(hidden.size()-1).tf.rank-1]));
		}
		in=ia.get("in");
		out=ia.get("out");
		Interface last=in;
		for(int i=0; i<hidden.size(); ++i){
			if(layers.get(i) instanceof InOutCastLayer)
				hidden.get(i).offset=last.offset;
			hidden.set(i, last=ia.allocate(hidden.get(i)));
			
		}
		String lastName="in";
		for(int i=0; i<layers.size(); ++i){
			String nextName = i>=hidden.size() ? "out" : hidden.get(i).name;
			layers.get(i).allocate(ia, lastName, nextName);
			lastName=nextName;
		}
		sc:for(int i=0; i<shortcutIn.size(); ++i){
			Interface si=shortcutIn.get(i);
			Interface so=i<shortcutOut.size()?shortcutOut.get(i):out;
			if(si==null)
				si=in;
			if(so==null)
				so=in;
			if(si==so) continue;
			int width = shortcutWidth.get(i);
			if(width==-1){
				if(si.count != so.count) continue;
				width=si.count;
			}else{
				if(si.tf != null || so.tf!=null){
					if(si.tf != null && so.tf != null){
						if(si.tf.dims.length != so.tf.dims.length)
							continue;
						for(int j=0; i<si.tf.dims.length-1; ++j){
							if(si.tf.dims[j] != so.tf.dims[j])
								continue sc;
							width*=si.tf.dims[j];
						}
					}else{
						TensorFormat tf = si.tf==null?si.tf:so.tf;
						for(int j=0; i<tf.dims.length-1; ++j)
							width*=tf.dims[j];
					}
				}
			}
			if(si.count<width) continue;
			if(so.count<width) continue;
			String siname, soname;
			if(si==in)siname="in";else if(si==out)siname="out"; else siname=si.name;
			if(so==in)soname="in";else if(so==out)soname="out"; else soname=so.name;
			
			Copy cl=new Copy(0, null);
			cl.width=width;
			cl.allocate(ia, siname, soname);
			shortcuts.add(cl);
		}
		for(int i=0, j=0; i<layers.size(); ++i) {
			InOutModule l = layers.get(i);
			exec.add(l);
			while(j<shortcuts.size() && shortcutIn.get(j)==l.getIn())
				exec.add(shortcuts.get(j++));
		}
		

	}

	@Override
	public void allocate(ParamAllocator pa) {
		for(Module m: layers){
			pa.push(null);
			m.allocate(pa);
			pa.pop();
		}
	}

	@Override
	public void share(ParamBlocks ps) {
		for(int i=0; i<layers.size(); ++i){
			layers.get(i).share(ps.get("Submodule "+i));
		}
	}

	@Override
	public ParamBlocks getParamBlocks() {
		ParamBlocks ret=new ParamBlocks("Stack");
		for(int i=0; i<layers.size(); ++i){
			ret.add("Submodule "+i, layers.get(i).getParamBlocks());
		}
		return null;
	}

	@Override
	public void forward(ForwardPhase training, ParamSet params, ActivationBatch as, int t, int[] inst) {
		for(Module m: exec) {
			m.forward(training, params, as, t, inst);
		}
	}

	@Override
	public void backprop(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, int t, int[] inst) {
		for(int i=exec.size()-1; i>=0; --i)
			exec.get(i).backprop(phase, params, as, errors, t, inst);
	}
	@Override
	public void gradients(String phase, ParamSet params, ActivationBatch as, ActivationBatch errors, ParamSet gradients,
			int t, int[] inst) {
		for(int i=exec.size()-1; i>=0; --i)
			exec.get(i).gradients(phase, params, as, errors, gradients, t, inst);
	}


////	@Override
//	public void declareDependencies(Dependencies d) {
////		for(Module l: layers)
////			l.declareDependencies(d);
//	}

	@Override
	public void dontComputeInPhase(String phase) {
		for(Interface h: hidden)
			h.dontComputeInPhase(phase);
	}
//	@Override
	public boolean wouldBackprop(String phase) {
		return true;
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
	public List<Module> getSubmodules() {
		return exec;
	}
}
