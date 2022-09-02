package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.Interface;

public interface InOutCastFactory {
	public abstract InOutCastLayer produce(Interface in);
}
