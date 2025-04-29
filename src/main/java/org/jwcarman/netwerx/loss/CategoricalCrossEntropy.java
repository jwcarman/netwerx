package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

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
    public double loss(SimpleMatrix yHat, SimpleMatrix y) {
        var loss = 0.0;

        for (int col = 0; col < yHat.getNumCols(); col++) {
            for (int row = 0; row < yHat.getNumRows(); row++) {
                var predicted = clamp(yHat.get(row, col), epsilon, 1.0 - epsilon);
                var actual = y.get(row, col);
                loss -= actual * Math.log(predicted);
            }
        }

        return loss / yHat.getNumCols();
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix yHat, SimpleMatrix y) {
        return yHat.minus(y);
    }

}
