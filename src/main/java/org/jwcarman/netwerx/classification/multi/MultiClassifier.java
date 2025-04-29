package org.jwcarman.netwerx.classification.multi;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.TrainingObserver;

public interface MultiClassifier {

    /**
     * Predicts the class index for each input column.
     */
    int[] predict(SimpleMatrix input);

    /**
     * Trains the classifier on the provided inputs and one-hot encoded labels.
     */
    void train(SimpleMatrix x, int[] labels, TrainingObserver observer);
}