package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.matrix.Matrix;

public interface Backprop<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    M a();

    M apply(M gradOutput);

}
