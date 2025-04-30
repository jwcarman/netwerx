package org.jwcarman.netwerx.classification.binary;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

public interface BinaryClassifier {

// -------------------------- STATIC METHODS --------------------------

    static BinaryClassifier create(NeuralNetwork network, Loss loss) {
        return new DefaultBinaryClassifier(network, loss);
    }

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the probability of each sample being in the positive class.
     *
     * @param samples A matrix of samples where each column is a sample and each row is a feature.
     * @return A row vector of probabilities for each sample.
     */
    boolean[] predict(SimpleMatrix samples);

    /**
     * Trains the binary classifier using the provided inputs and labels.
     *
     * @param inputs A matrix of inputs where each column is a sample and each row is a feature.
     * @param labels A boolean array where each element corresponds to a sample, indicating whether it belongs to the positive class (true) or negative class (false).
     * @param observer An observer to monitor the training process, which can be used to track progress, log information, or handle interruptions.
     */
    void train(SimpleMatrix inputs, boolean[] labels, TrainingObserver observer);

}
