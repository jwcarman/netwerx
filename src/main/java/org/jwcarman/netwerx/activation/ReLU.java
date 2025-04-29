package org.jwcarman.netwerx.activation;

public class ReLU extends HeActivation {

// --------------------------- CONSTRUCTORS ---------------------------

    public ReLU() {
    }

    public ReLU(double initialBias) {
        super(initialBias);
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double apply(double input) {
        return Math.max(0.0, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0.0 ? 1.0 : 0.0;
    }

}
