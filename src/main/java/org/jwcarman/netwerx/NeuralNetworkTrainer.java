package org.jwcarman.netwerx;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Interface for training a neural network.
 *
 * @param <M> the type of matrix used in the neural network
 */
public interface NeuralNetworkTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Trains a neural network using the provided training dataset.
     *
     * @param trainingDataset the dataset to train the neural network on
     * @return the trained neural network
     */
    NeuralNetwork<M> train(Dataset<M> trainingDataset);

}
