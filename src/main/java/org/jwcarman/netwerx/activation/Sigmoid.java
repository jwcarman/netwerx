package org.jwcarman.netwerx.activation;

public class Sigmoid extends ScalarActivationFunction {

// ------------------------------ FIELDS ------------------------------

    public static final ActivationFunction INSTANCE = new Sigmoid();

// --------------------------- CONSTRUCTORS ---------------------------

    private Sigmoid() {
        // singleton
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double derivative(double input) {
        var a = apply(input);
        return a * (1.0 - a);
    }

    @Override
    public double apply(double input) {
        final var clamped = Math.clamp(input, -500, 500);
        return 1.0 / (1.0 + Math.exp(-clamped));
    }

}
