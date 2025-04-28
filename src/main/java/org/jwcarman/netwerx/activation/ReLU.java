package org.jwcarman.netwerx.activation;

import java.util.Random;

public class ReLU extends ScalarActivation {

// ------------------------------ FIELDS ------------------------------

    private static final double DEFAULT_INITIAL_BIAS = 0.01;
    private final double initialBias;

// --------------------------- CONSTRUCTORS ---------------------------

    public ReLU() {
        this(DEFAULT_INITIAL_BIAS);
    }

    public ReLU(double initialBias) {
        this.initialBias = initialBias;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Activation ---------------------

    @Override
    public double generateInitialWeight(Random rand, int fanIn, int fanOut) {
        double stddev = Math.sqrt(2.0 / fanIn); // He Initialization
        return (rand.nextDouble() * 2 - 1) * stddev;
    }

    @Override
    public double generateInitialBias() {
        return initialBias;
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
