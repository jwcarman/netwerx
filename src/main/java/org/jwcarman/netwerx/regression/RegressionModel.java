package org.jwcarman.netwerx.regression;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.TrainingObserver;

public interface RegressionModel {

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
