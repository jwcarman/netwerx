package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.dataset.Dataset;
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
        var network = networkTrainer.train(new Dataset<>(inputs, inputs.multiClassifierOutputs(outputClasses, labels)), observer);
        return new DefaultMultiClassifier<>(network);
    }

}
