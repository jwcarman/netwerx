package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.Dataset;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public class DefaultMultiClassifierTrainer<M extends Matrix<M>> implements MultiClassifierTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetworkTrainer<M> networkTrainer;
    private final int outputClasses;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultMultiClassifierTrainer(NeuralNetworkTrainer<M> networkTrainer, int outputClasses) {
        this.networkTrainer = networkTrainer;
        this.outputClasses = outputClasses;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MultiClassifierTrainer ---------------------

    @Override
    public MultiClassifier<M> train(M inputs, int[] labels, TrainingObserver observer) {
        if (labels.length != inputs.columnCount()) {
            throw new IllegalArgumentException("Label count must match input column count.");
        }
        var network = networkTrainer.train(new Dataset<>(inputs, convertLabels(inputs, labels)), observer);
        return new DefaultMultiClassifier<>(network);
    }

// -------------------------- OTHER METHODS --------------------------

    private M convertLabels(M input, int[] labels) {
        return input.likeKind(outputClasses, labels.length)
                .map((row, col, value) -> labels[col] == row ? 1.0 : 0.0);
    }

}
