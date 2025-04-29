package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.util.Matrices;

/**
 * Log-Cosh Loss function.
 * A smooth approximation to Mean Absolute Error (MAE).
 */
public class LogCosh implements Loss {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        final var loss = Matrices.predictionTargets(predictions, targets)
                .mapToDouble(pt -> Math.log(Math.cosh(pt.prediction() - pt.target())))
                .sum();
        return loss / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());
        Matrices.predictionTargets(predictions, targets)
                .forEach(pt -> grad.set(pt.row(), pt.col(), Math.tanh(pt.prediction() - pt.target())));
        return grad;
    }

}
