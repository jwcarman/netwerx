package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.learning.LearningRateProvider;
import org.jwcarman.netwerx.learning.LearningRateProviders;
import org.jwcarman.netwerx.matrix.Matrix;

public class AdamOptimizer<M extends Matrix<M>> implements Optimizer<M> {

// ------------------------------ FIELDS ------------------------------

    private final LearningRateProvider learningRateProvider;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private M m; // First moment vector (mean)
    private M v; // Second moment vector (uncentered variance)
    private int t; // Time step

// --------------------------- CONSTRUCTORS ---------------------------

    public AdamOptimizer() {
        this(0.001, 0.9, 0.999, 1e-8);
    }

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        this(LearningRateProviders.constant(learningRate), beta1, beta2, epsilon);
    }

    public AdamOptimizer(LearningRateProvider learningRateProvider, double beta1, double beta2, double epsilon) {
        this.learningRateProvider = learningRateProvider;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public M optimize(int epoch, M param, M grad) {
        if (m == null) {
            m = grad.fill(0);
            v = grad.fill(0);
        }

        t++;

        // m = beta1 * m + (1 - beta1) * grad
        m = m.scale(beta1).add(grad.scale(1.0 - beta1));

        // v = beta2 * v + (1 - beta2) * (grad element-wise squared)
        var gradSquared = grad.elementMultiply(grad);
        v = v.scale(beta2).add(gradSquared.scale(1.0 - beta2));

        // Bias correction
        var mHat = m.elementDivide(1.0 - Math.pow(beta1, t));
        var vHat = v.elementDivide(1.0 - Math.pow(beta2, t));

        // param = param - learningRate * mHat / (sqrt(vHat) + epsilon)
        var update = mHat.elementDivide(vHat.elementPower(0.5).elementAdd(epsilon)).scale(learningRateProvider.getLearningRate(epoch));

        return param.subtract(update);
    }

}
