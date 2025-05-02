package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public interface BinaryClassifier<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the probability of each sample being in the positive class.
     *
     * @param samples A matrix of samples where each column is a sample and each row is a feature.
     * @return A row vector of probabilities for each sample.
     */
    boolean[] predict(M samples);

    /**
     * Trains the binary classifier using the provided inputs and labels.
     *
     * @param inputs A matrix of inputs where each column is a sample and each row is a feature.
     * @param labels A boolean array where each element corresponds to a sample, indicating whether it belongs to the positive class (true) or negative class (false).
     * @param observer An observer to monitor the training process, which can be used to track progress, log information, or handle interruptions.
     */
    void train(M inputs, boolean[] labels, OptimizerProvider<M> optimizerProvider, TrainingObserver observer);

}
