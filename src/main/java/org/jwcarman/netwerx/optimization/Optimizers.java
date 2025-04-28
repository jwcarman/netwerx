package org.jwcarman.netwerx.optimization;

public class Optimizers {

// -------------------------- STATIC METHODS --------------------------

    public static Optimizer sgd() {
        return new SgdOptimizer();
    }

    public static Optimizer sgd(double learningRate) {
        return new SgdOptimizer(learningRate);
    }

    public static Optimizer momentum() {
        return new MomentumOptimizer();
    }

    public static Optimizer momentum(double learningRate, double momentumFactor) {
        return new MomentumOptimizer(learningRate, momentumFactor);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Optimizers() {
        // Prevent instantiation
    }

}
