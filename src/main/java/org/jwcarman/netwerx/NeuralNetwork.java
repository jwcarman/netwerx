package org.jwcarman.netwerx;

import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.matrix.Matrix;
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

    /**
     * Trains the neural network using the provided input and target matrices.
     *
     * @param x        the input matrix, where each column represents a sample
     * @param y        the target output matrix, where each column corresponds to the expected output for the respective input sample
     * @param loss     the loss function to be used for training, which defines how the model's predictions are evaluated against the targets
     * @param observer an observer that can monitor the training process, providing feedback or logging information during training
     */
    void train(M x, M y, Loss loss, OptimizerProvider<M> optimizerProvider, TrainingObserver observer);

}
