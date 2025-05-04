package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.matrix.Matrix;

public class DefaultRegressionModel<M extends Matrix<M>> implements RegressionModel<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork<M> network;

// -------------------------- STATIC METHODS --------------------------


    public DefaultRegressionModel(NeuralNetwork<M> network) {
        this.network = network;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface RegressionModel ---------------------

    @Override
    public double[] predict(M input) {
        var output = network.predict(input);
        var predictions = new double[output.columnCount()];
        for (int col = 0; col < output.columnCount(); col++) {
            predictions[col] = output.valueAt(0, col);
        }
        return predictions;
    }
}
