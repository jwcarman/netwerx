package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.matrix.Matrix;

public class DefaultBinaryClassifier<M extends Matrix<M>> implements BinaryClassifier<M> {

// ------------------------------ FIELDS ------------------------------

    public static final double THRESHOLD = 0.5;
    private final NeuralNetwork<M> network;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultBinaryClassifier(NeuralNetwork<M> network) {
        this.network = network;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassifier ---------------------

    @Override
    public boolean[] predict(M input) {
        var probabilities = network.predict(input);
        var predictions = new boolean[probabilities.columnCount()];

        for (int col = 0; col < probabilities.columnCount(); col++) {
            predictions[col] = probabilities.valueAt(0, col) > THRESHOLD;
        }
        return predictions;
    }

}
