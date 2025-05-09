package org.jwcarman.netwerx.activation;

public class Tanh extends ScalarActivationFunction {

// ------------------------------ FIELDS ------------------------------

    public static final ActivationFunction INSTANCE = new Tanh();

// --------------------------- CONSTRUCTORS ---------------------------

    private Tanh() {
        // singleton
    }

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
