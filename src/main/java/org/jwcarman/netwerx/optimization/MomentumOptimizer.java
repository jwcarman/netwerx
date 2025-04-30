package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;

public class MomentumOptimizer implements Optimizer {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_LEARNING_RATE = 0.01;
    public static final double DEFAULT_MOMENTUM_FACTOR = 0.9;
    private final double learningRate;
    private final double momentumFactor;
    private SimpleMatrix velocity;

// --------------------------- CONSTRUCTORS ---------------------------

    public MomentumOptimizer() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM_FACTOR); // Default values
    }

    public MomentumOptimizer(double learningRate, double momentumFactor) {
        this.learningRate = learningRate;
        this.momentumFactor = momentumFactor;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public SimpleMatrix optimize(SimpleMatrix params, SimpleMatrix gradient) {
        if (velocity == null) {
            velocity = new SimpleMatrix(gradient.getNumRows(), gradient.getNumCols());
        }
        velocity = velocity.scale(momentumFactor).plus(gradient.scale(learningRate));
        return params.minus(velocity);
    }

}
