package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.layer.Layer;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.normalization.InputNormalizer;

import java.util.List;
import java.util.function.UnaryOperator;

class DefaultNeuralNetwork<M extends Matrix<M>> implements NeuralNetwork<M> {

// ------------------------------ FIELDS ------------------------------

    private final UnaryOperator<M> preprocessor;
    private final List<Layer<M>> layers;

// --------------------------- CONSTRUCTORS ---------------------------

    DefaultNeuralNetwork(UnaryOperator<M> preprocessor, List<Layer<M>> layers) {
        this.preprocessor = preprocessor;
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
        if (layers.getFirst().inputSize() != x.rowCount()) {
            throw new IllegalArgumentException("Input matrix row count does not match the first layer's input size.");
        }
        M input = preprocessor.apply(x);
        return layers.stream().reduce(input, (M acc, Layer<M> layer) -> layer.apply(acc), (a, _) -> a);
    }

    @Override
    public NeuralNetwork<M> headNetwork(int endIndex) {
        return new DefaultNeuralNetwork<>(preprocessor, layers.subList(0, endIndex));
    }

    @Override
    public NeuralNetwork<M> tailNetwork(int startIndex) {
        return new DefaultNeuralNetwork<>(InputNormalizer.empty(), layers.subList(startIndex, layers.size()));
    }

    @Override
    public List<Integer> layerSizes() {
        return layers.stream()
                .map(Layer::outputSize)
                .toList();
    }

}
