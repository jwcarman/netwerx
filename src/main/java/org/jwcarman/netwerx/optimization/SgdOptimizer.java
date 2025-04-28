package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;

import static org.jwcarman.netwerx.matrix.Matrices.subtractColumnVector;

public class SgdOptimizer implements Optimizer {

// ------------------------------ FIELDS ------------------------------

    private static final double DEFAULT_LEARNING_RATE = 0.01;

    private final double learningRate;

// --------------------------- CONSTRUCTORS ---------------------------

    public SgdOptimizer() {
        this(DEFAULT_LEARNING_RATE);
    }

    public SgdOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public SimpleMatrix optimize(SimpleMatrix parameter, SimpleMatrix gradient) {
        return subtractColumnVector(parameter, gradient.scale(learningRate));
    }
}
