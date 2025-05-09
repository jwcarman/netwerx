package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

public class DefaultBinaryClassifierTrainer<M extends Matrix<M>> implements BinaryClassifierTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetworkTrainer<M> networkTrainer;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultBinaryClassifierTrainer(NeuralNetworkTrainer<M> networkTrainer) {
        this.networkTrainer = networkTrainer;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassifierTrainer ---------------------

    @Override
    public BinaryClassifier<M> train(M features, boolean[] labels) {
        var dataset = Dataset.forBinaryClassifier(features, labels);
        var network = networkTrainer.train(dataset);
        return new DefaultBinaryClassifier<>(network);
    }

}
