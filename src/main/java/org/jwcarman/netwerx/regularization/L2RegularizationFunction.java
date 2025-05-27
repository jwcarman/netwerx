package org.jwcarman.netwerx.regularization;

import org.jwcarman.netwerx.matrix.Matrix;

public class L2RegularizationFunction<M extends Matrix<M>> implements RegularizationFunction<M> {

// ------------------------------ FIELDS ------------------------------

    private final double lambda;

// --------------------------- CONSTRUCTORS ---------------------------

    public L2RegularizationFunction(double lambda) {
        this.lambda = lambda;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Regularizer ---------------------

    @Override
    public M gradient(M matrix) {
        return matrix.scale(lambda);
    }

    @Override
    public double penalty(M matrix) {
        return lambda / 2.0 * matrix.sumOfSquares(); // or normL2()^2
    }

}
