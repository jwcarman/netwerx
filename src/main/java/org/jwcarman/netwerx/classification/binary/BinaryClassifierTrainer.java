package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.observer.TrainingObservers;

public interface BinaryClassifierTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Trains a binary classifier using the provided features and labels.
     * @param features the input features for training where each column represents a sample and each row represents a feature.
     * @param labels the binary labels for each sample, where true represents one class and false represents the other.
     * @return a BinaryClassifier instance trained on the provided features and labels.
     */
    default BinaryClassifier<M> train(M features, boolean[] labels) {
        return train(features, labels, TrainingObservers.noop());
    }

    /**
     * Trains a binary classifier using the provided features and labels.
     *
     * @param features the input features for training where each column represents a sample and each row represents a feature.
     * @param labels   the binary labels for each sample, where true represents one class and false represents the other.
     * @param observer an observer to monitor the training process.
     * @return a BinaryClassifier instance trained on the provided features and labels.
     */
    BinaryClassifier<M> train(M features, boolean[] labels, TrainingObserver observer);

}
