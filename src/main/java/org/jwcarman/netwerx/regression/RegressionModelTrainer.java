package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.matrix.Matrix;

public interface RegressionModelTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Train a regression model using the provided features and labels.
     *
     * @param inputs the feature matrix (shape: features x samples)
     * @param labels the target values (shape: samples)
     * @return the trained regression model
     */
    RegressionModel<M> train(M inputs, double[] labels);

}
