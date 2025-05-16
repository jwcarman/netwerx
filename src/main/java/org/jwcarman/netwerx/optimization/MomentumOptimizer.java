package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.learning.LearningRateProvider;
import org.jwcarman.netwerx.learning.LearningRateProviders;
import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Momentum optimizer (Polyak, 1964).
 * <p>
 * Updates parameters using momentum-based gradient descent:
 * v = μ * v + η * ∇θ
 * θ = θ - v
 * <p>
 * Where:
 * μ = momentum factor
 * η = learning rate
 */
public class MomentumOptimizer<M extends Matrix<M>> implements Optimizer<M> {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_LEARNING_RATE = 0.01;
    public static final double DEFAULT_MOMENTUM_FACTOR = 0.9;
    private final LearningRateProvider learningRateProvider;
    private final double momentumFactor;
    private M velocity;

// --------------------------- CONSTRUCTORS ---------------------------

    public MomentumOptimizer() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM_FACTOR); // Default values
    }

    public MomentumOptimizer(double learningRateProvider, double momentumFactor) {
        this(LearningRateProviders.constant(learningRateProvider), momentumFactor);
    }

    public MomentumOptimizer(LearningRateProvider learningRateProvider, double momentumFactor) {
        this.learningRateProvider = learningRateProvider;
        this.momentumFactor = momentumFactor;
    }

    // ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public M optimize(int epoch, M params, M gradient) {
        if (velocity == null) {
            velocity = gradient.fill(0.0); // Initialize velocity to zero
        }
        velocity = velocity.scale(momentumFactor).add(gradient.scale(learningRateProvider.getLearningRate(epoch)));
        return params.subtract(velocity);
    }

}
