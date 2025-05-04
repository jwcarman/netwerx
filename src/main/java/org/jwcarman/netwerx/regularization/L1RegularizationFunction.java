package org.jwcarman.netwerx.regularization;

import org.jwcarman.netwerx.matrix.Matrix;

public class L1RegularizationFunction<M extends Matrix<M>> implements RegularizationFunction<M> {

// ------------------------------ FIELDS ------------------------------

    private final double lambda;

// --------------------------- CONSTRUCTORS ---------------------------

    public L1RegularizationFunction(double lambda) {
        this.lambda = lambda;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Regularizer ---------------------

    @Override
    public M gradient(M matrix) {
        return matrix.map((_, _, v) -> lambda * Math.signum(v));
    }

    @Override
    public double penalty(M matrix) {
        return lambda * matrix.sumOfAbs();
    }

}
