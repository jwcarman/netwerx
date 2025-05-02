package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Log-Cosh Loss function.
 * A smooth approximation to Mean Absolute Error (MAE).
 */
public class LogCosh implements Loss {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------


    @Override
    public <M extends Matrix<M>> M gradient(M predictions, M targets) {
        return predictions.map((row, col, yHat) -> {
            var y = targets.valueAt(row, col);
            var diff = yHat - y;
            return Math.tanh(diff);
        });
    }

    @Override
    public <M extends Matrix<M>> double loss(M predictions, M targets) {
        final var loss = Losses.predictionTargets(predictions, targets)
                .mapToDouble(pt -> Math.log(Math.cosh(pt.prediction() - pt.target())))
                .sum();
        return loss / predictions.size();
    }

}
