package org.jwcarman.netwerx.classification.multi;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

public interface MultiClassifier {

// -------------------------- STATIC METHODS --------------------------

    static MultiClassifier create(NeuralNetwork network, Loss loss, int outputClasses) {
        return new DefaultMultiClassifier(network, loss, outputClasses);
    }

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the class index for each input column.
     */
    int[] predict(SimpleMatrix input);

    /**
     * Trains the classifier on the provided inputs and one-hot encoded labels.
     */
    void train(SimpleMatrix x, int[] labels, TrainingObserver observer);

}
