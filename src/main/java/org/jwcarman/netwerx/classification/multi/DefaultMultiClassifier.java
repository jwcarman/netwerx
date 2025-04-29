package org.jwcarman.netwerx.classification.multi;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

public class DefaultMultiClassifier implements MultiClassifier {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork network;
    private final Loss loss;
    private final int outputClasses;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultMultiClassifier(NeuralNetwork network, Loss loss, int outputClasses) {
        this.network = network;
        this.loss = loss;
        this.outputClasses = outputClasses;
    }

// ------------------------ INTERFACE METHODS ------------------------

    @Override
    public int[] predict(SimpleMatrix input) {
        var probabilities = network.predict(input);
        int[] predictions = new int[probabilities.getNumCols()];

        for (int col = 0; col < probabilities.getNumCols(); col++) {
            int maxIndex = 0;
            double maxValue = probabilities.get(0, col);

            for (int row = 1; row < probabilities.getNumRows(); row++) {
                if (probabilities.get(row, col) > maxValue) {
                    maxValue = probabilities.get(row, col);
                    maxIndex = row;
                }
            }

            predictions[col] = maxIndex;
        }

        return predictions;
    }

    @Override
    public void train(SimpleMatrix x, int[] labels, TrainingObserver observer) {
        network.train(x, convertLabels(labels), loss, observer);
    }

    private SimpleMatrix convertLabels(int[] labels) {
        var matrix = new SimpleMatrix(outputClasses, labels.length);
        for (int i = 0; i < labels.length; i++) {
            matrix.set(labels[i], i, 1.0);
        }
        return matrix;
    }
}