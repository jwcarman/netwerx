package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.util.Matrices;

public class Huber implements Loss {

    private final double delta;

    public Huber() {
        this(1.0);
    }

    public Huber(double delta) {
        this.delta = delta;
    }

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        final var sum = Matrices.predictionTargets(predictions, targets)
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
        return sum / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());
        Matrices.predictionTargets(predictions, targets)
                .forEach(pt -> {
                    var diff = pt.prediction() - pt.target();
                    var absDiff = Math.abs(diff);
                    double value;
                    if (absDiff <= delta) {
                        value = diff;
                    } else {
                        value = delta * Math.signum(diff);
                    }
                    grad.set(pt.row(), pt.col(), value);
                });
        return grad.divide(1.0 * predictions.getNumRows() * predictions.getNumCols());
    }
}