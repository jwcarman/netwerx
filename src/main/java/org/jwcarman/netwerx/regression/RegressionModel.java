package org.jwcarman.netwerx.regression;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

public interface RegressionModel {

// -------------------------- STATIC METHODS --------------------------

    static RegressionModel create(NeuralNetwork network, Loss loss) {
        return new DefaultRegressionModel(network, loss);
    }

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predict the outputs for a batch of inputs.
     *
     * @param inputs the feature matrix (shape: features x samples)
     * @return the predicted outputs as an array of doubles
     */
    double[] predict(SimpleMatrix inputs);

    /**
     * Train the model on the given inputs and labels.
     *
     * @param inputs the feature matrix (features x samples)
     * @param labels the target values (1D array of doubles)
     * @param observer a training observer for early stopping
     */
    void train(SimpleMatrix inputs, double[] labels, TrainingObserver observer);

}
