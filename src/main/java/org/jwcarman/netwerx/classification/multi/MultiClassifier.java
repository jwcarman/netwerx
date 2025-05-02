package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public interface MultiClassifier<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the class index for each input column.
     */
    int[] predict(M input);

    /**
     * Trains the classifier on the provided inputs and one-hot encoded labels.
     */
    void train(M x, int[] labels, OptimizerProvider<M> optimizerProvider, TrainingObserver observer);

}
