package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.matrix.Matrix;

public interface RegressionModel<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predict the outputs for a batch of inputs.
     *
     * @param inputs the feature matrix (shape: features x samples)
     * @return the predicted outputs as an array of doubles
     */
    double[] predict(M inputs);
}
