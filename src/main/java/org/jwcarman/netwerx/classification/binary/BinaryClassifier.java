package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.matrix.Matrix;

public interface BinaryClassifier<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the label for each input column of the sample matrix.
     *
     * @param samples the input matrix containing samples as columns
     * @return an array of boolean values where each value corresponds to the predicted label for each sample
     */
    boolean[] predictLabels(M samples);

}
