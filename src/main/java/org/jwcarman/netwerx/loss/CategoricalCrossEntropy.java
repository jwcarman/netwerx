package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.util.Matrices;

import static java.lang.Math.clamp;

public class CategoricalCrossEntropy implements Loss {

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
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        var loss = Matrices.predictionTargets(predictions, targets)
                .mapToDouble(pt -> {
                    var yHat = clamp(pt.prediction(), epsilon, 1.0 - epsilon);
                    var y = pt.target();
                    return -y * Math.log(yHat);
                })
                .sum();
        return loss / predictions.getNumCols();
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix yHat, SimpleMatrix y) {
        return yHat.minus(y);
    }

}
