package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

public class WeightedBinaryCrossEntropy implements Loss {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_EPSILON = 1e-15;
    private final double positiveWeight; // Weight for label 1
    private final double negativeWeight; // Weight for label 0
    private final double epsilon;

// --------------------------- CONSTRUCTORS ---------------------------

    public WeightedBinaryCrossEntropy(double positiveWeight, double negativeWeight) {
        this(positiveWeight, negativeWeight, DEFAULT_EPSILON);
    }

    public WeightedBinaryCrossEntropy(double positiveWeight, double negativeWeight, double epsilon) {
        this.positiveWeight = positiveWeight;
        this.negativeWeight = negativeWeight;
        this.epsilon = epsilon;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------


    @Override
    public <M extends Matrix<M>> M gradient(M a, M y) {
        return a.subtract(y); // Gradient shape same regardless of weighting
    }

    @Override
    public <M extends Matrix<M>> double loss(M a, M y) {
        var clamped = a.clamp(epsilon, 1 - epsilon);

        var term1 = y.elementMultiply(clamped.log()).scale(positiveWeight);

        var oneMinusY = y.scale(-1).elementAdd(1);
        var oneMinusA = clamped.scale(-1).elementAdd(1);
        var term2 = oneMinusY.elementMultiply(oneMinusA.log()).scale(negativeWeight);

        return -(term1.add(term2)).sum() / a.size();
    }

}
