package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

public class Huber implements Loss {

    private final double delta;

    public Huber() {
        this(1.0);
    }

    public Huber(double delta) {
        this.delta = delta;
    }

    @Override
    public <M extends Matrix<M>> double loss(M predictions, M targets) {
        final var sum = Losses.predictionTargets(predictions, targets)
                .mapToDouble(pt -> {
                    var diff = pt.prediction() - pt.target();
                    var absDiff = Math.abs(diff);
                    if (absDiff <= delta) {
                        return 0.5 * diff * diff;
                    } else {
                        return delta * (absDiff - 0.5 * delta);
                    }
                })
                .sum();
        return sum / predictions.size();
    }

    @Override
    public <M extends Matrix<M>> M gradient(M predictions, M targets) {
        return predictions.map((row, col, yHat) -> {
            var y = targets.valueAt(row, col);
            var diff = yHat - y;
            var absDiff = Math.abs(diff);
            if (absDiff <= delta) {
                return diff / predictions.size();
            } else {
                return delta * Math.signum(diff) / predictions.size();
            }
        });
    }
}