package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.matrix.Matrices;

public class WeightedBinaryCrossEntropy implements LossFunction {

// ------------------------------ FIELDS ------------------------------

    private final double positiveWeight; // Weight for label 1
    private final double negativeWeight; // Weight for label 0
    private final double epsilon;

// --------------------------- CONSTRUCTORS ---------------------------

    public WeightedBinaryCrossEntropy(double positiveWeight, double negativeWeight) {
        this(positiveWeight, negativeWeight, 1e-15);
    }

    public WeightedBinaryCrossEntropy(double positiveWeight, double negativeWeight, double epsilon) {
        this.positiveWeight = positiveWeight;
        this.negativeWeight = negativeWeight;
        this.epsilon = epsilon;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LossFunction ---------------------

    @Override
    public double loss(SimpleMatrix a, SimpleMatrix y) {
        var clamped = Matrices.clamp(a, epsilon, 1 - epsilon);

        var term1 = y.elementMult(clamped.elementLog()).scale(positiveWeight);
        var term2 = (y.scale(-1).plus(1))
                .elementMult(clamped.scale(-1).plus(1).elementLog())
                .scale(negativeWeight);

        return -(term1.plus(term2)).elementSum() / (a.getNumCols() * a.getNumRows());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix a, SimpleMatrix y) {
        return a.minus(y); // Gradient shape same regardless of weighting
    }

}
