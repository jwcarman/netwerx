package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.def.DefaultNeuralNetworkBuilder;
import org.jwcarman.netwerx.loss.Loss;

/**
 * Interface representing a neural network model.
 */
public interface NeuralNetwork {

// -------------------------- STATIC METHODS --------------------------

    /**
     * Creates a new instance of {@link NeuralNetworkBuilder} to construct a neural network.
     *
     * @param inputSize the size of the input layer, which corresponds to the number of features in the input data
     * @return a new instance of {@link NeuralNetworkBuilder} for building the neural network
     */
    static NeuralNetworkBuilder builder(int inputSize) {
        return new DefaultNeuralNetworkBuilder(inputSize);
    }

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the output for a given input matrix.
     *
     * @param x the input matrix, where each column represents a sample
     * @return the predicted output matrix, where each column corresponds to the prediction for the respective input sample
     */
    SimpleMatrix predict(SimpleMatrix x);

    /**
     * Trains the neural network using the provided input and target matrices.
     *
     * @param x        the input matrix, where each column represents a sample
     * @param y        the target output matrix, where each column corresponds to the expected output for the respective input sample
     * @param loss     the loss function to be used for training, which defines how the model's predictions are evaluated against the targets
     * @param observer an observer that can monitor the training process, providing feedback or logging information during training
     */
    void train(SimpleMatrix x, SimpleMatrix y, Loss loss, TrainingObserver observer);

}
