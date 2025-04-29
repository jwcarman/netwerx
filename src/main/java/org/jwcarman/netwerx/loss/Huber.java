package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

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
        double sum = 0.0;
        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                double diff = predictions.get(row, col) - targets.get(row, col);
                double absDiff = Math.abs(diff);
                if (absDiff <= delta) {
                    sum += 0.5 * diff * diff;
                } else {
                    sum += delta * (absDiff - 0.5 * delta);
                }
            }
        }
        return sum / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());
        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                double diff = predictions.get(row, col) - targets.get(row, col);
                if (Math.abs(diff) <= delta) {
                    grad.set(row, col, diff);
                } else {
                    grad.set(row, col, Math.signum(diff) * delta);
                }
            }
        }
        return grad.divide(1.0 * predictions.getNumRows() * predictions.getNumCols());
    }
}