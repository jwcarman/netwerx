package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.matrix.Matrix;

public interface MultiClassifier<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the class index for each input column.
     */
    int[] predict(M input);

}
