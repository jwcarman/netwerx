package org.jwcarman.netwerx.activation;

public class Activations {

// -------------------------- STATIC METHODS --------------------------

    public static Activation sigmoid() {
        return new Sigmoid();
    }

    public static Activation relu() {
        return new ReLU();
    }

    public static Activation relu(double initialBias) {
        return new ReLU(initialBias);
    }

    public static Activation tanh() {
        return new Tanh();
    }

    public static Activation leakyRelu() {
        return new LeakyReLU();
    }

    public static Activation leakyRelu(double initialBias, double alpha) {
        return new LeakyReLU(initialBias, alpha);
    }

    public static Activation softmax() {
        return new Softmax();
    }

    public static Activation linear() {
        return new Linear();
    }

    public static Activation swish() {
        return new Swish();
    }

    public static Activation swish(double initialBias) {
        return new Swish(initialBias);
    }

    public static Activation elu() {
        return new ELU();
    }

    public static Activation elu(double initialBias, double alpha) {
        return new ELU(initialBias, alpha);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Activations() {
        // Prevent instantiation
    }

}
