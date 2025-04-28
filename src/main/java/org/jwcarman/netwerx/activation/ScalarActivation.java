package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;

public abstract class ScalarActivation implements Activation {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ActivationFunction ---------------------

    public SimpleMatrix apply(SimpleMatrix input) {
        return input.elementOp((SimpleOperations.ElementOpReal) (_, _, v) -> apply(v));
    }

    public SimpleMatrix derivative(SimpleMatrix input) {
        return input.elementOp((SimpleOperations.ElementOpReal) (_, _, v) -> derivative(v));
    }

// -------------------------- OTHER METHODS --------------------------

    protected abstract double apply(double x);

    protected abstract double derivative(double x);

}
