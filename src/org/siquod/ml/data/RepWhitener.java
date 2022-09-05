package org.siquod.ml.data;

public class RepWhitener implements Whitener{
	public static interface RepIndexer{
		public int index(int repIndex, int channelIndex);
	}
	public final Whitener base;
	final double[] inBufferD;
	final double[] outBufferD;
	final float[] inBufferF;
	final float[] outBufferF;
	public final int repetitions;
	public final RepIndexer indexer;
	final int chs;
	public RepWhitener(Whitener base, int repetitions, RepIndexer repi) {
		this.base=base;
		this.repetitions = repetitions;
		this.indexer=repi;
		chs = this.base.dim();
		this.inBufferD = new double[chs];
		this.outBufferD = new double[base.dim()];
		this.inBufferF = new float[chs];
		this.outBufferF = new float[base.dim()];
	}
	@Override
	public void whiten(double[] in, double[] out) {
		for(int r=0; r<repetitions; ++r) {
			for(int c=0; c<chs; ++c)
				inBufferD[c] = in[indexer.index(r, c)]; 
			base.whiten(inBufferD, outBufferD);
			for(int c=0; c<chs; ++c)
				out[indexer.index(r, c)] = outBufferD[c];
		}
	}
	@Override
	public void whiten(float[] in, float[] out) {
		for(int r=0; r<repetitions; ++r) {
			for(int c=0; c<chs; ++c)
				inBufferF[c] = in[indexer.index(r, c)]; 
			base.whiten(inBufferF, outBufferF);
			for(int c=0; c<chs; ++c)
				out[indexer.index(r, c)] = outBufferF[c]; 
		}
	}
	@Override
	public int dim() {
		return chs*repetitions;
	}
	
}
