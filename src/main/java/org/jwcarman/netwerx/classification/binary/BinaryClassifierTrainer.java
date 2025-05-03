package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public interface BinaryClassifierTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    default BinaryClassifier<M> train(M inputs, boolean[] labels) {
        return train(inputs, labels, TrainingObserver.noop());
    }

    BinaryClassifier<M> train(M inputs, boolean[] labels, TrainingObserver observer);

}
