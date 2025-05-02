package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.matrix.Matrix;

/**
 * RMSProp optimizer.
 * <p>
 * Reference: Tieleman & Hinton (2012).
 */
public class RmsPropOptimizer<M extends Matrix<M>> implements Optimizer<M> {

// ------------------------------ FIELDS ------------------------------

    private final double learningRate;
    private final double beta;
    private final double epsilon;

    private M v; // Moving average of squared gradients

// --------------------------- CONSTRUCTORS ---------------------------

    public RmsPropOptimizer() {
        this(0.001, 0.9, 1e-8);
    }

    public RmsPropOptimizer(double learningRate, double beta, double epsilon) {
        this.learningRate = learningRate;
        this.beta = beta;
        this.epsilon = epsilon;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Optimizer ---------------------

    @Override
    public M optimize(M param, M grad) {
        if (v == null) {
            v = grad.fill(0.0);
        }

        // v = beta * v + (1 - beta) * grad^2
        v = v.scale(beta).add(grad.elementMultiply(grad).scale(1.0 - beta));

        // param = param - learningRate * grad / (sqrt(v) + epsilon)
        var update = grad.elementDivide(v.elementPower(0.5).elementAdd(epsilon)).scale(learningRate);

        return param.subtract(update);
    }

}
