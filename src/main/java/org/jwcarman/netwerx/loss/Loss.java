package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

public interface Loss {

// -------------------------- OTHER METHODS --------------------------

    <M extends Matrix<M>> M gradient(M output, M target);

    <M extends Matrix<M>> double loss(M output, M target);

}
