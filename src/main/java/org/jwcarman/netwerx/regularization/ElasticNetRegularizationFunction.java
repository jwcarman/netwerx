package org.jwcarman.netwerx.regularization;

import org.jwcarman.netwerx.matrix.Matrix;

public class ElasticNetRegularizationFunction<M extends Matrix<M>> implements RegularizationFunction<M> {

// ------------------------------ FIELDS ------------------------------

    private final double lambda1;
    private final double lambda2;

// --------------------------- CONSTRUCTORS ---------------------------

    public ElasticNetRegularizationFunction(double lambda1, double lambda2) {
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface RegularizationFunction ---------------------

    @Override
    public M gradient(M matrix) {
        return matrix.map((_, _, v) -> lambda1 * Math.signum(v) + lambda2 * 2 * v);
    }

    @Override
    public double penalty(M matrix) {
        return lambda1 * matrix.sumOfAbs() + lambda2 * matrix.sumOfSquares();
    }

}
