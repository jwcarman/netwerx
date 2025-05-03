package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public interface MultiClassifierTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    default MultiClassifier<M> train(M x, int[] labels) {
        return train(x, labels, TrainingObserver.noop());
    }

    MultiClassifier<M> train(M x, int[] labels, TrainingObserver observer);

}
