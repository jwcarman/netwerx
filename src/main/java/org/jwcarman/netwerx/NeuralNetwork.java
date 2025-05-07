package org.jwcarman.netwerx;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.List;

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

    NeuralNetwork<M> subNetwork(int startIndex, int endIndex);

    NeuralNetwork<M> subNetwork(int startIndex);

    List<Integer> layerSizes();
}
