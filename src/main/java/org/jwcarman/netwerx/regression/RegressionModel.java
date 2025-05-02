package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public interface RegressionModel<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predict the outputs for a batch of inputs.
     *
     * @param inputs the feature matrix (shape: features x samples)
     * @return the predicted outputs as an array of doubles
     */
    double[] predict(M inputs);

    /**
     * Train the model on the given inputs and labels.
     *
     * @param inputs the feature matrix (features x samples)
     * @param labels the target values (1D array of doubles)
     * @param observer a training observer for early stopping
     */
    void train(M inputs, double[] labels, OptimizerProvider<M> optimizerProvider, TrainingObserver observer);

}
