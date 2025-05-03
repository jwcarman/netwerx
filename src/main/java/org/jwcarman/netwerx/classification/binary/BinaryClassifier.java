package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.matrix.Matrix;

public interface BinaryClassifier<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the probability of each sample being in the positive class.
     *
     * @param samples A matrix of samples where each column is a sample and each row is a feature.
     * @return A row vector of probabilities for each sample.
     */
    boolean[] predict(M samples);

}
