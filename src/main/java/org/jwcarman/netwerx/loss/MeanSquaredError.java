package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

public class MeanSquaredError implements Loss {

    @Override
    public double loss(SimpleMatrix predictions, SimpleMatrix targets) {
        var diff = predictions.minus(targets);
        return diff.elementPower(2).elementSum() / (predictions.getNumCols() * predictions.getNumRows());
    }

    @Override
    public SimpleMatrix gradient(SimpleMatrix predictions, SimpleMatrix targets) {
        return predictions.minus(targets).scale(2.0 / (predictions.getNumCols() * predictions.getNumRows()));
    }
}