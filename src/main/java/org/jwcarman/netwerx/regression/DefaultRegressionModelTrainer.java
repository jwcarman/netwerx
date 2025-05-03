package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public class DefaultRegressionModelTrainer<M extends Matrix<M>> implements RegressionModelTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetworkTrainer<M> networkTrainer;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultRegressionModelTrainer(NeuralNetworkTrainer<M> networkTrainer) {
        this.networkTrainer = networkTrainer;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface RegressionModelTrainer ---------------------

    @Override
    public RegressionModel<M> train(M inputs, double[] labels, TrainingObserver observer) {
        if (labels.length != inputs.columnCount()) {
            throw new IllegalArgumentException("Label count must match input row count.");
        }
        var dataset = new Dataset<>(inputs, convertLabels(inputs, labels));
        var network = networkTrainer.train(dataset, observer);
        return new DefaultRegressionModel<>(network);
    }

// -------------------------- OTHER METHODS --------------------------

    private M convertLabels(M inputs, double[] labels) {
        return inputs.likeKind(1, inputs.columnCount()).map((_, col, _) -> labels[col]);
    }

}
