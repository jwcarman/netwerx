package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.matrix.Matrices;

public class BinaryCrossEntropy implements LossFunction {
    public static final double DEFAULT_EPSILON = 1e-15;

// ------------------------------ FIELDS ------------------------------

    private final double epsilon;

// --------------------------- CONSTRUCTORS ---------------------------

    public BinaryCrossEntropy() {
        this(DEFAULT_EPSILON);
    }

    public BinaryCrossEntropy(double epsilon) {
        this.epsilon = epsilon; // Allow custom epsilon value
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LossFunction ---------------------

    @Override
    public double loss(SimpleMatrix a, SimpleMatrix y) {
        var clamped = Matrices.clamp(a, epsilon, 1 - epsilon);

        SimpleMatrix term1 = y.elementMult(clamped.elementLog());
        SimpleMatrix term2 = y.scale(-1).plus(1)
                .elementMult(clamped.scale(-1).plus(1).elementLog());

        return - (term1.plus(term2)).elementSum() / (a.getNumCols() * a.getNumRows());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix a, SimpleMatrix y) {
        return a.minus(y);
    }

}
