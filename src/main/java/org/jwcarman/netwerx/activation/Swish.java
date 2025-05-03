package org.jwcarman.netwerx.activation;

public class Swish extends HeInitializedActivationFunction {

// --------------------------- CONSTRUCTORS ---------------------------

    public Swish() {
    }

    public Swish(double initialBias) {
        super(initialBias);
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
