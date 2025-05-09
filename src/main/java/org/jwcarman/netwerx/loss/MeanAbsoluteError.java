package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

public class MeanAbsoluteError implements LossFunction {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------


    @Override
    public <M extends Matrix<M>> M gradient(M predictions, M targets) {
        return predictions.map((row, col, yHat) -> {
            var y = targets.valueAt(row, col);
            var diff = yHat - y;
            return Math.signum(diff) / predictions.size();
        });
    }

    @Override
    public <M extends Matrix<M>> double loss(M predictions, M targets) {
        final var sum = PredictionTarget.of(predictions, targets)
                .mapToDouble(pt -> Math.abs(pt.prediction() - pt.target()))
                .sum();

        return sum / predictions.size();
    }

}
