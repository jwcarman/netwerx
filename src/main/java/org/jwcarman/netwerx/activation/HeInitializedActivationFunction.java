package org.jwcarman.netwerx.activation;

import java.util.Random;

import static org.jwcarman.netwerx.distribution.Distributions.heUniform;

public abstract class HeInitializedActivationFunction extends ScalarActivationFunction {

// ------------------------------ FIELDS ------------------------------

    private static final double DEFAULT_INITIAL_BIAS = 0.01;
    private final double initialBias;

// --------------------------- CONSTRUCTORS ---------------------------

    protected HeInitializedActivationFunction() {
        this(DEFAULT_INITIAL_BIAS);
    }

    protected HeInitializedActivationFunction(double initialBias) {
        this.initialBias = initialBias;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ActivationFunction ---------------------

    @Override
    public double initialBias(Random rand, int fanIn, int fanOut) {
        return initialBias;
    }

    @Override
    public double initialWeight(Random rand, int fanIn, int fanOut) {
        return heUniform(rand, fanIn);
    }

}
