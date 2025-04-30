package org.jwcarman.netwerx.regression;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

public class DefaultRegressionModel implements RegressionModel{

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork network;
    private final Loss loss;

// -------------------------- STATIC METHODS --------------------------

// -------------------------- HELPER METHODS --------------------------
    private static SimpleMatrix convertLabels(double[] labels) {
        var numSamples = labels.length;
        var y = new SimpleMatrix(1, numSamples);
        for (int col = 0; col < numSamples; col++) {
            y.set(0, col, labels[col]);
        }
        return y;
    }

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultRegressionModel(NeuralNetwork network, Loss loss) {
        this.network = network;
        this.loss = loss;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface RegressionModel ---------------------

    @Override
    public double[] predict(SimpleMatrix input) {
        var output = network.predict(input);
        var predictions = new double[output.getNumCols()];
        for (int col = 0; col < output.getNumCols(); col++) {
            predictions[col] = output.get(0, col);
        }
        return predictions;
    }

    @Override
    public void train(SimpleMatrix inputs, double[] labels, TrainingObserver observer) {
        network.train(inputs, convertLabels(labels), loss, observer);
    }

}
