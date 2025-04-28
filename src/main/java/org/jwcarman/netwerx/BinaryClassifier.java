package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;

public interface BinaryClassifier {

    /**
     * Predicts the probability of each sample being in the positive class.
     *
     * @param samples A matrix of samples where each column is a sample and each row is a feature.
     * @return A row vector of probabilities for each sample.
     */
    SimpleMatrix predict(SimpleMatrix samples);

    /**
     * Trains the binary classifier using the provided inputs and targets.
     * @param inputs the input features for training, where each column is a sample and each row is a feature.
     * @param targets a row vector of target values, where each value is either 0 or 1 indicating the class label for each sample.
     * @param observer an observer that can be notified of training progress, losses, and other events.
     */
    void train(SimpleMatrix inputs, SimpleMatrix targets, TrainingObserver observer);

}
