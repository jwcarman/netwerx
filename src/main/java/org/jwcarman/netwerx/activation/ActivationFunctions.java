package org.jwcarman.netwerx.activation;

public class ActivationFunctions {

// -------------------------- STATIC METHODS --------------------------

    public static ActivationFunction sigmoid() {
        return new Sigmoid();
    }

    public static ActivationFunction relu() {
        return new ReLU();
    }

    public static ActivationFunction relu(double initialBias) {
        return new ReLU(initialBias);
    }

    public static ActivationFunction tanh() {
        return new Tanh();
    }

    public static ActivationFunction leakyRelu() {
        return new LeakyReLU();
    }

    public static ActivationFunction leakyRelu(double initialBias, double alpha) {
        return new LeakyReLU(initialBias, alpha);
    }

    public static ActivationFunction softmax() {
        return new Softmax();
    }

    public static ActivationFunction linear() {
        return new Linear();
    }

    public static ActivationFunction swish() {
        return new Swish();
    }

    public static ActivationFunction swish(double initialBias) {
        return new Swish(initialBias);
    }

    public static ActivationFunction elu() {
        return new ELU();
    }

    public static ActivationFunction elu(double initialBias, double alpha) {
        return new ELU(initialBias, alpha);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private ActivationFunctions() {
        // Prevent instantiation
    }

}
