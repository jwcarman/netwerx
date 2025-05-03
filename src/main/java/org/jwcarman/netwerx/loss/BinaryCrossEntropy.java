package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

public class BinaryCrossEntropy implements LossFunction {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_EPSILON = 1e-15;
    private final double epsilon;

// --------------------------- CONSTRUCTORS ---------------------------

    public BinaryCrossEntropy() {
        this(DEFAULT_EPSILON);
    }

    public BinaryCrossEntropy(double epsilon) {
        this.epsilon = epsilon; // Allow custom epsilon value
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------


    @Override
    public <M extends Matrix<M>> M gradient(M a, M y) {
        return a.subtract(y);
    }

    @Override
    public <M extends Matrix<M>> double loss(M a, M y) {
        var clamped = a.clamp(epsilon, 1 - epsilon);

        M term1 = y.elementMultiply(clamped.log());
        M term2 = y.negate().elementAdd(1)
                .elementMultiply(clamped.negate().elementAdd(1).log());

        return -(term1.add(term2)).sum() / (a.size());
    }

}
