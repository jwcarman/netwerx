package org.jwcarman.netwerx.activation;

public class Swish extends ScalarActivationFunction {

// ------------------------------ FIELDS ------------------------------

    public static final ActivationFunction INSTANCE = new Swish();

// --------------------------- CONSTRUCTORS ---------------------------

    private Swish() {
        // singleton
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double apply(double x) {
        return x / (1.0 + Math.exp(-x)); // x * sigmoid(x)
    }

    @Override
    public double derivative(double x) {
        double sigmoid = 1.0 / (1.0 + Math.exp(-x));
        return sigmoid + x * sigmoid * (1.0 - sigmoid);
    }

}
