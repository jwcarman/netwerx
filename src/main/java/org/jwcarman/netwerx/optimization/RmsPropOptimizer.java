package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;

/**
 * RMSProp optimizer.
 *
 * Reference: Tieleman & Hinton (2012).
 */
public class RmsPropOptimizer implements Optimizer {

// ------------------------------ FIELDS ------------------------------

    private final double learningRate;
    private final double beta;
    private final double epsilon;

    private SimpleMatrix v; // Moving average of squared gradients

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
    public SimpleMatrix optimize(SimpleMatrix param, SimpleMatrix grad) {
        if (v == null) {
            v = new SimpleMatrix(grad.getNumRows(), grad.getNumCols());
        }

        // v = beta * v + (1 - beta) * grad^2
        v = v.scale(beta).plus(grad.elementMult(grad).scale(1.0 - beta));

        // param = param - learningRate * grad / (sqrt(v) + epsilon)
        var update = grad.elementDiv(v.elementPower(0.5).plus(epsilon)).scale(learningRate);

        return param.minus(update);
    }

}
