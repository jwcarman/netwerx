package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

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
    public RegressionModel<M> train(M inputs, double[] targets) {
        var dataset = Dataset.forRegressionModel(inputs, targets);
        var network = networkTrainer.train(dataset);
        return new DefaultRegressionModel<>(network);
    }

}
