package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

/**
 * Log-Cosh Loss function.
 * A smooth approximation to Mean Absolute Error (MAE).
 */
public class LogCosh implements Loss {

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        double loss = 0.0;
        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                double diff = predictions.get(row, col) - targets.get(row, col);
                loss += Math.log(Math.cosh(diff));
            }
        }
        return loss / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());
        for (int row = 0; row < predictions.getNumRows(); row++) {
            for (int col = 0; col < predictions.getNumCols(); col++) {
                double diff = predictions.get(row, col) - targets.get(row, col);
                grad.set(row, col, Math.tanh(diff));
            }
        }
        return grad;
    }
}