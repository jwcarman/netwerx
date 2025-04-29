package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.util.Matrices;

public class MeanAbsoluteError implements Loss {

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        final var sum = Matrices.predictionTargets(predictions, targets)
                .mapToDouble(pt -> Math.abs(pt.prediction() - pt.target()))
                .sum();

        return sum / (predictions.getNumRows() * predictions.getNumCols());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        var grad = new SimpleMatrix(predictions.getNumRows(), predictions.getNumCols());
        Matrices.predictionTargets(predictions, targets)
                .forEach(pt -> grad.set(pt.row(), pt.col(), Math.signum(pt.prediction() - pt.target())));
        return grad.divide(1.0 * predictions.getNumRows() * predictions.getNumCols());
    }
}