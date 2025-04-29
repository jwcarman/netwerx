package org.jwcarman.netwerx.activation;

import java.util.Random;

public abstract class HeActivation extends ScalarActivation {

// ------------------------------ FIELDS ------------------------------

    private static final double DEFAULT_INITIAL_BIAS = 0.01;
    private final double initialBias;

// --------------------------- CONSTRUCTORS ---------------------------

    protected HeActivation() {
        this(DEFAULT_INITIAL_BIAS);
    }

    protected HeActivation(double initialBias) {
        this.initialBias = initialBias;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Activation ---------------------

    @Override
    public double generateInitialWeight(Random rand, int fanIn, int fanOut) {
        double stddev = Math.sqrt(2.0 / fanIn);
        return (rand.nextDouble() * 2 - 1) * stddev;
    }

    @Override
    public double generateInitialBias() {
        return initialBias;
    }

}
