package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.layer.Layer;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.List;

class DefaultNeuralNetwork<M extends Matrix<M>> implements NeuralNetwork<M> {

// ------------------------------ FIELDS ------------------------------

    private final List<Layer<M>> layers;

// --------------------------- CONSTRUCTORS ---------------------------

    DefaultNeuralNetwork(List<Layer<M>> layers) {
        this.layers = layers;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetwork ---------------------

    /**
     * Predicts the output for the given input using the neural network.
     *
     * @param x the input features as a matrix (each column is a feature vector)
     * @return the predicted output as a matrix (each column is a predicted output vector)
     */
    @Override
    public M predict(M x) {
        if(layers.getFirst().inputSize() != x.rowCount()) {
            throw new IllegalArgumentException("Input matrix row count does not match the first layer's input size.");
        }
        return layers.stream().reduce(x, (M acc, Layer<M> layer) -> layer.apply(acc), (a, _) -> a);
    }

    @Override
    public NeuralNetwork<M> subNetwork(int startIndex, int endIndex) {
        return new DefaultNeuralNetwork<>(layers.subList(startIndex, endIndex));
    }

    @Override
    public NeuralNetwork<M> subNetwork(int startIndex) {
        return new DefaultNeuralNetwork<>(layers.subList(startIndex, layers.size()));
    }

    @Override
    public List<Integer> layerSizes() {
        return layers.stream()
                .map(Layer::outputSize)
                .toList();
    }

}
