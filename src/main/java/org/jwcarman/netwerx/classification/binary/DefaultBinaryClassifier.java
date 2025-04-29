package org.jwcarman.netwerx.classification.binary;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

import java.util.stream.IntStream;

public class DefaultBinaryClassifier implements BinaryClassifier {

// ------------------------------ FIELDS ------------------------------

    public static final double TRUE = 1.0;
    public static final double FALSE = 0.0;
    public static final double THRESHOLD = 0.5;
    private final NeuralNetwork network;
    private final Loss loss;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultBinaryClassifier(NeuralNetwork network, Loss loss) {
        this.network = network;
        this.loss = loss;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassifier ---------------------

    @Override
    public boolean[] predict(SimpleMatrix input) {
        var probabilities = network.predict(input);
        var predictions = new boolean[probabilities.getNumCols()];

        for (int col = 0; col < probabilities.getNumCols(); col++) {
            predictions[col] = probabilities.get(0, col) > THRESHOLD;
        }

        return predictions;
    }

    @Override
    public void train(SimpleMatrix x, boolean[] labels, TrainingObserver observer) {
        network.train(x, convertLabels(labels), loss, observer);
    }

    private static SimpleMatrix convertLabels(boolean[] labels) {
        var numSamples = labels.length;
        var y = new SimpleMatrix(1, numSamples);
        IntStream.range(0, numSamples).forEach(i -> y.set(0, i, labels[i] ? TRUE : FALSE));
        return y;
    }
}
