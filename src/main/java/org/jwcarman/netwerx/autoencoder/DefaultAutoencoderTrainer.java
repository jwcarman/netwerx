package org.jwcarman.netwerx.autoencoder;

import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

import java.util.List;

public class DefaultAutoencoderTrainer<M extends Matrix<M>> implements AutoencoderTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetworkTrainer<M> networkTrainer;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultAutoencoderTrainer(NeuralNetworkTrainer<M> networkTrainer) {
        this.networkTrainer = networkTrainer;
    }

    private static int findBottleneckIndex(List<Integer> layerSizes) {
        int minIdx = -1;
        int minSize = Integer.MAX_VALUE;
        for (int i = 0; i < layerSizes.size(); i++) {
            int size = layerSizes.get(i);
            if (size < minSize) {
                minSize = size;
                minIdx = i;
            }
        }
        return minIdx;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface AutoencoderTrainer ---------------------

    @Override
    public Autoencoder<M> train(M input, TrainingObserver observer) {
        var dataset = Dataset.forAutoencoder(input);
        var network = networkTrainer.train(dataset, observer);
        var bottleneckIndex = findBottleneckIndex(network.layerSizes());
        var encoder = network.subNetwork(0, bottleneckIndex + 1);
        var decoder = network.subNetwork(bottleneckIndex + 1);
        return new DefaultAutoencoder<>(encoder, decoder);
    }

}
