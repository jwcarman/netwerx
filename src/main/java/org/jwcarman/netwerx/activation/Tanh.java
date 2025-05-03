package org.jwcarman.netwerx.activation;

public class Tanh extends ScalarActivationFunction {

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double apply(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double input) {
        var a = Math.tanh(input);
        return 1.0 - a * a;
    }

}
