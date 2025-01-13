package org.siquod.ml.data;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class GenericFileCursor implements TrainingBatchCursor.RandomAccess, AutoCloseable{

	File inputFile;
	long entryCount;
	long position=0;
	InputStream fromFile;
	int output;
	double[] fixedOutput;
	int inputFloatsPerSet;
	double[] inputValues; 
	final double[] outputValues; 
	/**
	 * If this is null, the weight is read from the file, after the outputs
	 */
	Boolean normalizeWeights;
	final int weightFloatsPerSet;
	int oneHot;
	final int outputFloatsPerSet;
	final int outputFloatsInFile;
	double weight;

	public GenericFileCursor(File file, int output, int oneHot, int floatsPerSet, Boolean normalizeWeights) {
		inputFile=file;
		this.output=output;
		this.inputFloatsPerSet=floatsPerSet;
		inputValues = new double[floatsPerSet];
		this.normalizeWeights=normalizeWeights;
		this.oneHot = oneHot;
		outputFloatsPerSet=fixedOutput!=null?fixedOutput.length:oneHot==0?1:oneHot;
		outputFloatsInFile=0;
		outputValues=null;
		weightFloatsPerSet = normalizeWeights==null?1:0;
		reset();
	}
	public GenericFileCursor(File file, int inputFloatsPerSet, int outputFloatsPerSet, Boolean normalizeWeights) {
		inputFile=file;
		this.output=-1;
		this.inputFloatsPerSet=inputFloatsPerSet;
		inputValues = new double[inputFloatsPerSet];
		this.normalizeWeights=normalizeWeights;
		this.oneHot = -1;
		this.outputFloatsPerSet=outputFloatsPerSet;
		outputFloatsInFile=outputFloatsPerSet;
		outputValues=new double[outputFloatsPerSet];
		weightFloatsPerSet = normalizeWeights==null?1:0;
		reset();
	}
	public GenericFileCursor(File file, double[] fixedOutput, int floatsPerSet, Boolean normalizeWeights) {
		inputFile=file;
		this.output=0;
		this.inputFloatsPerSet=floatsPerSet;
		inputValues = new double[floatsPerSet];
		this.normalizeWeights=normalizeWeights;
		this.oneHot = -1;
		this.fixedOutput=fixedOutput;
		outputFloatsPerSet=fixedOutput.length;
		outputFloatsInFile=0;
		outputValues=null;
		weightFloatsPerSet = normalizeWeights==null?1:0;
		reset();
	}

	@Override
	public void reset() {
		try {
			if(fromFile!=null)
				fromFile.close();
			fromFile=new BufferedInputStream(new FileInputStream(inputFile));
			// entryCount = total bytes/ (#datapoints in set * # bytes in datapoint)
			entryCount=inputFile.length()/((inputFloatsPerSet+outputFloatsInFile+weightFloatsPerSet)*4);
			position=0;
			if (entryCount>0)
				loadNextDataPoint();
		} catch (IOException e){
			e.printStackTrace();
		}
	}

	@Override
	public GenericFileCursor clone() {
		return new GenericFileCursor(inputFile, output, oneHot, inputFloatsPerSet, normalizeWeights);
	}

	@Override
	public double getWeight() {
		if(normalizeWeights==null) {
			return weight;
		}
		return normalizeWeights?1d/entryCount:1; 
	}

	@Override
	public void next() {
		// position just needed to know if we're at end of file
		position+=1; // each datapoint (red,green,blue)
		if (!isFinished())
			loadNextDataPoint();
	}
	private void loadNextDataPoint() {
		try {
			for (int i=0; i<inputValues.length; i++) {
				inputValues[i]=readFloat(fromFile);
			}
			for (int i=0; i<outputFloatsInFile; i++) {
				outputValues[i]=readFloat(fromFile);
			}  
			if(weightFloatsPerSet==1) {
				weight = readFloat(fromFile);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static float readFloat(InputStream under) throws IOException{
		int bits=readInt(under);
		float f = Float.intBitsToFloat(bits);
		//		if(Double.isNaN(f))
		//			System.out.println();
		return f;
	}
	public static int readInt(InputStream under) throws IOException{
		return 
				(((int)under.read())<<24) |
				(((int)under.read())<<16) |
				(((int)under.read())<< 8) |
				(((int)under.read())    );
	}
	@Override
	public boolean isFinished() {
		if (position>=entryCount) // if position reaches last entryCount (> incase next()
			// 2x called because isFinished() checked
			return true;          // finish going through file
		return false;
	}

	@Override
	public void giveInputs(double[] inputs) {
		System.arraycopy(inputValues, 0, inputs, 0, inputValues.length);
	}

	@Override
	public void giveOutputs(double[] outputs) {
		if(outputValues!=null) {
			System.arraycopy(outputValues, 0, outputs, 0, outputFloatsInFile);
		}else if(fixedOutput!=null) {
			System.arraycopy(fixedOutput, 0, outputs, 0, fixedOutput.length);
		}else if(oneHot==0) {
			outputs[0]=output;
		}else {
			Arrays.fill(outputs, 0, oneHot, 0);
			outputs[output] = 1;
		}
	}

	//    public void giveOutputs0Or1(int outputs) { // ?? bc param should be int not double[] rite?
	//    	//WHAT???
	//        if (fromFile.equals("nonBoundaryFile"))
	//            output=0;
	//        else if (fromFile.equals("boundaryFile"))
	//            output=1;
	//    }

	@Override
	public int inputCount() {
		return inputFloatsPerSet;
	}

	@Override
	public int outputCount() {
		return outputFloatsPerSet;
	}

	@Override
	public void close() throws IOException { 
		fromFile.close();  
	}

	@Override
	public long size() {
		return entryCount;
	}

	@Override
	public void seek(long position) {
		if(position<this.position) {
			reset();
		}
		if(position>this.position) {
			long toSkip = (position-this.position - 1)*4*(inputFloatsPerSet+outputFloatsInFile+weightFloatsPerSet);
			try {
				while(toSkip>0) {
					long skipped = fromFile.skip(toSkip);
					if(skipped==0) {
						fromFile.read();
						++skipped;
					}
					toSkip-=skipped;
				}
				loadNextDataPoint();
				this.position=position;
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}
	public TrainingBatchCursor.RandomAccess ramBufferAndClose() {
		RamBuffer ret = ramBuffer();
		try {
			close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ret;
	}
}
