package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.observer.TrainingObservers;

public interface BinaryClassifierTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    default BinaryClassifier<M> train(M inputs, boolean[] labels) {
        return train(inputs, labels, TrainingObservers.noop());
    }

    BinaryClassifier<M> train(M inputs, boolean[] labels, TrainingObserver observer);

}
