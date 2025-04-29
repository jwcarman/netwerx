package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.util.Matrices;

/**
 * Hinge Loss function (for binary classification).
 * Assumes labels are -1 or +1.
 */
public class Hinge implements Loss {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        final var loss = Matrices.predictionTargets(predictions, targets)
                .mapToDouble(pt -> Math.max(0.0, 1.0 - pt.target() * pt.prediction()))
                .sum();
        return loss / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());
        Matrices.predictionTargets(predictions, targets)
                .forEach(pt -> {
                    final var y = pt.target();
                    final var yHat = pt.prediction();
                    grad.set(pt.row(), pt.col(), y * yHat < 1.0 ? -y : 0.0);
                });
        return grad.divide(1.0 * predictions.getNumRows() * predictions.getNumCols());
    }

}
