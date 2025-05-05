package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.matrix.Matrix;

public class DefaultMultiClassifier<M extends Matrix<M>> implements MultiClassifier<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork<M> network;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultMultiClassifier(NeuralNetwork<M> network) {
        this.network = network;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MultiClassifier ---------------------

    @Override
    public int[] predictClasses(M input) {
        var argMax = network.predict(input).columnArgMax();
        int[] predictions = new int[argMax.columnCount()];
        for (int i = 0; i < predictions.length; i++) {
            predictions[i] = (int) argMax.valueAt(0, i);
        }
        return predictions;
    }

}
