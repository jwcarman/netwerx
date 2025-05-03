package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public interface MultiClassifier<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the class index for each input column.
     */
    int[] predict(M input);

}
