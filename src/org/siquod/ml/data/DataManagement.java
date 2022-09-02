package org.siquod.ml.data;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class DataManagement {
	public static <E> List<E> concatExcept(List<List<E>> parts, int except){
		List<E> ret = new ArrayList<>();
		for(int i=0; i<parts.size(); ++i)
			if(i!=except)
				ret.addAll(parts.get(i));
			else
				System.out.println("dropped some samples from the training set: "+parts.get(i).size());
		return ret;
	}
	public static <E> List<List<E>> split(List<E> data, boolean shuffle, int parts){
		double[] ratios=new double[parts];
		Arrays.fill(ratios, 1);
		return split(data, shuffle, ratios);
	}

	public static <E> List<List<E>> split(List<E> data, boolean shuffle, double... ratios){
		List<List<E>> ret = new ArrayList<>(ratios.length);
		double ratioSum=0;
		for(int i=0; i<ratios.length; ++i)
			ratioSum+=ratios[i];
		double renorm = data.size()/ratioSum;

		int[] endIndices=new int[ratios.length];
		{
			double lastEndIndex=0;
			for(int i=0; i<ratios.length; ++i) {
				double nextEndIndex=lastEndIndex + renorm*ratios[i];
				endIndices[i]=(int)Math.round(nextEndIndex);
				lastEndIndex=nextEndIndex;
			}
			endIndices[ratios.length-1]=data.size();
		}
		if(shuffle) {
			List<E> ndata=new ArrayList<>(data);
			for(int i=0; i<data.size(); ++i) {
				int j = i + (int)(Math.random()*(data.size()-i));
				if(i!=j) {
					E t = ndata.get(i);
					ndata.set(i, ndata.get(j));
					ndata.set(j, t);
				}
			}
			data=ndata;
		}
		int lastEndIndex=0;
		for(int i=0; i<ratios.length; ++i) {
			int nextEndIndex=endIndices[i];
			List<E> relem = new ArrayList<>(nextEndIndex-lastEndIndex);
			for(int j=lastEndIndex; j<nextEndIndex; ++j)
				relem.add(data.get(j));
			ret.add(relem);
			lastEndIndex=nextEndIndex;
		}
		return ret;
	}
	public static <E> List<List<E>> split(List<E> data, Random rand, int parts){
		double[] ratios=new double[parts];
		Arrays.fill(ratios, 1);
		return split(data, rand, ratios);
	}

	public static <E> List<List<E>> split(List<E> data, Random rand, double... ratios){
		List<List<E>> ret = new ArrayList<>(ratios.length);
		double ratioSum=0;
		for(int i=0; i<ratios.length; ++i)
			ratioSum+=ratios[i];
		double renorm = data.size()/ratioSum;

		int[] endIndices=new int[ratios.length];
		{
			double lastEndIndex=0;
			for(int i=0; i<ratios.length; ++i) {
				double nextEndIndex=lastEndIndex + renorm*ratios[i];
				endIndices[i]=(int)Math.round(nextEndIndex);
				lastEndIndex=nextEndIndex;
			}
			endIndices[ratios.length-1]=data.size();
		}
		if(rand!=null) {
			List<E> ndata=new ArrayList<>(data);
			for(int i=0; i<data.size(); ++i) {
				int j = i + (int)(rand.nextInt(data.size()-i));
				if(i!=j) {
					E t = ndata.get(i);
					ndata.set(i, ndata.get(j));
					ndata.set(j, t);
				}
			}
			data=ndata;
		}
		int lastEndIndex=0;
		for(int i=0; i<ratios.length; ++i) {
			int nextEndIndex=endIndices[i];
			List<E> relem = new ArrayList<>(nextEndIndex-lastEndIndex);
			for(int j=lastEndIndex; j<nextEndIndex; ++j)
				relem.add(data.get(j));
			ret.add(relem);
			lastEndIndex=nextEndIndex;
		}
		return ret;
	}
	public static void main(String[] args) {
		List<String> l = new ArrayList<>();
		for(char c='A'; c<='Z'; ++c)
			l.add(String.valueOf(c));
		System.out.println(split(l, true, 14));
	}
	public static Random cloneRandom(Random rand) {
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(bos);
			oos.writeObject(rand);
			oos.close();
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			ObjectInputStream ois = new ObjectInputStream(bis);
			return (Random)ois.readObject();
		}catch(IOException|ClassNotFoundException x) {
			x.printStackTrace();
		}
		return null;
	}

}

