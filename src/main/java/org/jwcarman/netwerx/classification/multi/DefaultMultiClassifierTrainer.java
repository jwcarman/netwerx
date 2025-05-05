package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
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
    public MultiClassifier<M> train(M features, int[] classes, TrainingObserver observer) {
        var dataset = Dataset.forMultiClassifier(features, outputClasses, classes);
        var network = networkTrainer.train(dataset, observer);
        return new DefaultMultiClassifier<>(network);
    }

}
