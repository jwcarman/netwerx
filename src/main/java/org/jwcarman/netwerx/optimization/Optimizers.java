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

    public static Optimizer adam() {
        return new AdamOptimizer();
    }

    public static Optimizer adam(double learningRate, double beta1, double beta2, double epsilon) {
        return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Optimizers() {
        // Prevent instantiation
    }

}
