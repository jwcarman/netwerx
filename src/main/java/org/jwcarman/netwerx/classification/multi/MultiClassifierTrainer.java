package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.observer.TrainingObservers;

public interface MultiClassifierTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    default MultiClassifier<M> train(M x, int[] labels) {
        return train(x, labels, TrainingObservers.noop());
    }

    MultiClassifier<M> train(M x, int[] labels, TrainingObserver observer);

}
