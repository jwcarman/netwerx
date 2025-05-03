package org.jwcarman.netwerx.activation;

public class Linear extends ScalarActivationFunction {

// -------------------------- OTHER METHODS --------------------------

    @Override
    protected double apply(double input) {
        return input;
    }

    @Override
    protected double derivative(double input) {
        return 1.0;
    }

}
