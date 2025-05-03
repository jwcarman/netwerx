package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;

public class MeanSquaredError implements LossFunction {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Loss ---------------------


    @Override
    public <M extends Matrix<M>> M gradient(M predictions, M targets) {
        return predictions.subtract(targets).scale(2.0 / predictions.size());
    }

    @Override
    public <M extends Matrix<M>> double loss(M predictions, M targets) {
        var diff = predictions.subtract(targets);
        return diff.elementPower(2).sum() / predictions.size();
    }

}
