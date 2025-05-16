package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.learning.LearningRateProvider;
import org.jwcarman.netwerx.learning.LearningRateProviders;
import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Standard Stochastic Gradient Descent (SGD) optimizer.
 * <p>
 * Updates parameters using:
 * θ = θ - η * ∇θ
 * <p>
 * No momentum or adaptive behavior — simple and efficient.
 */
public class SgdOptimizer<M extends Matrix<M>> implements Optimizer<M> {

// ------------------------------ FIELDS ------------------------------

    private static final double DEFAULT_LEARNING_RATE = 0.01;

    private final LearningRateProvider learningRateProvider;

// --------------------------- CONSTRUCTORS ---------------------------

    public SgdOptimizer() {
        this(DEFAULT_LEARNING_RATE);
    }

    public SgdOptimizer(double learningRate) {
        this(LearningRateProviders.constant(learningRate));
    }

    public SgdOptimizer(LearningRateProvider learningRateProvider) {
        this.learningRateProvider = learningRateProvider;
    }

    // ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public M optimize(int epoch, M parameter, M gradient) {
        return parameter.subtract(gradient.scale(learningRateProvider.getLearningRate(epoch)));
    }

}
