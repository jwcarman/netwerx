package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

/**
 * Hinge Loss function (for binary classification).
 * Assumes labels are -1 or +1.
 */
public class Hinge implements Loss {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        double loss = 0.0;
        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                double y = targets.get(row, col);
                double yHat = predictions.get(row, col);
                loss += Math.max(0.0, 1.0 - y * yHat);
            }
        }
        return loss / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());

        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                double y = targets.get(row, col);
                double yHat = predictions.get(row, col);
                grad.set(row, col, y * yHat < 1.0 ? -y : 0.0);
            }
        }
        return grad.divide(1.0 * predictions.getNumRows() * predictions.getNumCols());
    }

}
