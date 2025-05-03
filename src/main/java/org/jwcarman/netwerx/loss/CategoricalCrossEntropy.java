package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

import static java.lang.Math.clamp;

public class CategoricalCrossEntropy implements LossFunction {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_EPSILON = 1e-15;
    private final double epsilon;

// --------------------------- CONSTRUCTORS ---------------------------

    public CategoricalCrossEntropy() {
        this(DEFAULT_EPSILON);
    }

    public CategoricalCrossEntropy(double epsilon) {
        this.epsilon = epsilon;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------


    @Override
    public <M extends Matrix<M>> M gradient(M yHat, M y) {
        return yHat.subtract(y);
    }

    @Override
    public <M extends Matrix<M>> double loss(M predictions, M targets) {
        var loss = PredictionTarget.of(predictions, targets)
                .mapToDouble(pt -> {
                    var yHat = clamp(pt.prediction(), epsilon, 1.0 - epsilon);
                    var y = pt.target();
                    return -y * Math.log(yHat);
                })
                .sum();
        return loss / predictions.columnCount();
    }

}
