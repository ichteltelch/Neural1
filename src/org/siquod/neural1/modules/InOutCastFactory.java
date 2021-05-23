package org.siquod.neural1.modules;

import org.siquod.neural1.Interface;

public interface InOutCastFactory {
	public abstract InOutCastLayer produce(Interface in);
}
