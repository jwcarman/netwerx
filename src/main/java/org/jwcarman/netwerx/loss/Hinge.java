package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Hinge Loss function (for binary classification).
 * Assumes labels are -1 or +1.
 */
public class Hinge implements Loss {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------

    @Override
    public <M extends Matrix<M>> double loss(M predictions, M targets) {
        final var loss = PredictionTarget.of(predictions, targets)
                .mapToDouble(pt -> Math.max(0.0, 1.0 - pt.target() * pt.prediction()))
                .sum();
        return loss / predictions.size();
    }

    @Override
    public <M extends Matrix<M>> M gradient(M predictions, M targets) {
        return predictions.map((row, col, yHat) -> {
            final var y = targets.valueAt(row, col);
            return (y * yHat < 1.0 ? -y : 0.0) / predictions.size();
        });
    }

}
