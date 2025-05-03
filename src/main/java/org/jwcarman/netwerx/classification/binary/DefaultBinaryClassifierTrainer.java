package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public class DefaultBinaryClassifierTrainer<M extends Matrix<M>> implements BinaryClassifierTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private static final double TRUE = 1.0;
    private static final double FALSE = 0.0;

    private final NeuralNetworkTrainer<M> networkTrainer;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultBinaryClassifierTrainer(NeuralNetworkTrainer<M> networkTrainer) {
        this.networkTrainer = networkTrainer;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassifierTrainer ---------------------

    @Override
    public BinaryClassifier<M> train(M inputs, boolean[] labels, TrainingObserver observer) {
        if (labels.length != inputs.columnCount()) {
            throw new IllegalArgumentException("Label count must match input row count.");
        }
        var network = networkTrainer.train(new Dataset<>(inputs, convertLabels(inputs, labels)), observer);
        return new DefaultBinaryClassifier<>(network);
    }

// -------------------------- OTHER METHODS --------------------------

    private M convertLabels(M input, boolean[] labels) {
        return input.likeKind(1, input.columnCount()).map((row, col, value) -> labels[col] ? TRUE : FALSE);
    }

}
