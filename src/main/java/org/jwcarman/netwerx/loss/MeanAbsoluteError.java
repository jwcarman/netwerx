package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

public class MeanAbsoluteError implements Loss {

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        double sum = 0.0;
        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                sum += Math.abs(predictions.get(row, col) - targets.get(row, col));
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
                grad.set(row, col, Math.signum(diff));
            }
        }
        return grad;
    }
}