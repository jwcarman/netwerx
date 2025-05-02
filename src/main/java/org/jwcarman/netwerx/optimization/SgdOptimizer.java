package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Standard Stochastic Gradient Descent (SGD) optimizer.
 * <p>
 * Updates parameters using:
 *   θ = θ - η * ∇θ
 * <p>
 * No momentum or adaptive behavior — simple and efficient.
 */
public class SgdOptimizer<M extends Matrix<M>> implements Optimizer<M> {

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
    public M optimize(M parameter, M gradient) {
        return parameter.subtract(gradient.scale(learningRate));
    }

}
