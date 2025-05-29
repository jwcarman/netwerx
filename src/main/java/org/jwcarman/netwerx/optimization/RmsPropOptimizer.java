package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.learning.LearningRateProvider;
import org.jwcarman.netwerx.learning.LearningRateProviders;
import org.jwcarman.netwerx.matrix.Matrix;

/**
 * RMSProp optimizer.
 * <p>
 * Reference: Tieleman &amp; Hinton (2012).
 */
public class RmsPropOptimizer<M extends Matrix<M>> implements Optimizer<M> {

// ------------------------------ FIELDS ------------------------------

    private final LearningRateProvider learningRateProvider;
    private final double beta;
    private final double epsilon;

    private M v; // Moving average of squared gradients

// --------------------------- CONSTRUCTORS ---------------------------

    public RmsPropOptimizer() {
        this(0.001, 0.9, 1e-8);
    }

    public RmsPropOptimizer(double learningRate, double beta, double epsilon) {
        this(LearningRateProviders.constant(learningRate), beta, epsilon);
    }

    public RmsPropOptimizer(LearningRateProvider learningRateProvider, double beta, double epsilon) {
        this.learningRateProvider = learningRateProvider;
        this.beta = beta;
        this.epsilon = epsilon;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public M optimize(int epoch, M param, M grad) {
        if (v == null) {
            v = grad.fill(0.0);
        }

        // v = beta * v + (1 - beta) * grad^2
        v = v.scale(beta).add(grad.elementMultiply(grad).scale(1.0 - beta));

        // param = param - learningRate * grad / (sqrt(v) + epsilon)
        var update = grad.elementDivide(v.elementPower(0.5).elementAdd(epsilon)).scale(learningRateProvider.getLearningRate(epoch));

        return param.subtract(update);
    }

}
