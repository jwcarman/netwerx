package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public class DefaultRegressionModel<M extends Matrix<M>> implements RegressionModel<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork<M> network;
    private final Loss loss;

// -------------------------- STATIC METHODS --------------------------

// -------------------------- HELPER METHODS --------------------------
    private M convertLabels(M inputs, double[] labels) {
        return inputs.likeKind(1, inputs.columnCount()).map((_, col, _) -> labels[col]);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultRegressionModel(NeuralNetwork<M> network, Loss loss) {
        this.network = network;
        this.loss = loss;
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

    @Override
    public void train(M inputs, double[] labels, OptimizerProvider<M> optimizerProvider, TrainingObserver observer) {
        network.train(inputs, convertLabels(inputs, labels), loss, optimizerProvider, observer);
    }

}
