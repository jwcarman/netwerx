package org.jwcarman.netwerx.activation;

public class ActivationFunctions {

// -------------------------- STATIC METHODS --------------------------

    public static ActivationFunction sigmoid() {
        return Sigmoid.INSTANCE;
    }

    public static ActivationFunction relu() {
        return ReLU.INSTANCE;
    }

    public static ActivationFunction tanh() {
        return Tanh.INSTANCE;
    }

    public static ActivationFunction leakyRelu() {
        return LeakyReLU.DEFAULT;
    }

    public static ActivationFunction leakyRelu(double alpha) {
        return new LeakyReLU(alpha);
    }

    public static ActivationFunction softmax() {
        return Softmax.INSTANCE;
    }

    public static ActivationFunction linear() {
        return Linear.INSTANCE;
    }

    public static ActivationFunction swish() {
        return Swish.INSTANCE;
    }

    public static ActivationFunction elu() {
        return ELU.DEFAULT;
    }

    public static ActivationFunction elu(double alpha) {
        return new ELU(alpha);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private ActivationFunctions() {
        // Prevent instantiation
    }

}
