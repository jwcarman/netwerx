package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

public abstract class ScalarActivationFunction implements ActivationFunction {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Activation ---------------------

    public <M extends Matrix<M>> M apply(M input) {
        return input.map((row, col, v) -> apply(v));
    }

    public <M extends Matrix<M>> M derivative(M input) {
        return input.map((row, col, v) -> derivative(v));
    }

// -------------------------- OTHER METHODS --------------------------

    protected abstract double apply(double x);

    protected abstract double derivative(double x);

}
