package org.jwcarman.netwerx.regularization;

import org.jwcarman.netwerx.matrix.Matrix;

public class NoopRegularizationFunction<M extends Matrix<M>> implements RegularizationFunction<M> {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Regularizer ---------------------

    @Override
    public M gradient(M matrix) {
        return matrix.fill(0);
    }

    @Override
    public double penalty(M matrix) {
        return 0.0;
    }

}
