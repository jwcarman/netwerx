package org.jwcarman.netwerx;

import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

/**
 * Interface representing a neural network model.
 */
public interface NeuralNetwork<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the output for a given input matrix.
     *
     * @param x the input matrix, where each column represents a sample
     * @return the predicted output matrix, where each column corresponds to the prediction for the respective input sample
     */
    M predict(M x);

}
