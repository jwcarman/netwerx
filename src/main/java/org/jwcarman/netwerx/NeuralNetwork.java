package org.jwcarman.netwerx;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.List;

/**
 * Interface representing a neural network model.
 */
public interface NeuralNetwork<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Gets the sizes (number of neurons) of each layer in the network.
     *
     * @return a list of integers representing the sizes of each layer
     */
    List<Integer> layerSizes();

    /**
     * Predicts the output for a given input matrix.
     *
     * @param x the input matrix, where each column represents a sample
     * @return the predicted output matrix, where each column corresponds to the prediction for the respective input sample
     */
    M predict(M x);

    /**
     * Creates a subnetwork starting from the specified index to the end of the network.
     *
     * @param startIndex the index of the first layer to include in the subnetwork
     * @return a new NeuralNetwork instance representing the subnetwork
     */
    NeuralNetwork<M> subNetwork(int startIndex);

    /**
     * Creates a subnetwork from the specified start index to the end index (exclusive).
     *
     * @param startIndex the index of the first layer to include in the subnetwork
     * @param endIndex   the index of the last layer to include in the subnetwork (exclusive)
     * @return a new NeuralNetwork instance representing the subnetwork
     */
    NeuralNetwork<M> subNetwork(int startIndex, int endIndex);

}
