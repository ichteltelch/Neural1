package org.siquod.ml.neural1.modules;

import org.siquod.ml.neural1.Interface;

public interface InOutFactory {
	InOutModule produce(Interface in, Interface out);
}
