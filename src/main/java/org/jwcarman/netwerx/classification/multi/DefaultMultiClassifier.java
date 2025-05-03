package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

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
    public int[] predict(M input) {
        var probabilities = network.predict(input);
        int[] predictions = new int[probabilities.columnCount()];

        for (int col = 0; col < probabilities.columnCount(); col++) {
            int maxIndex = 0;
            double maxValue = probabilities.valueAt(0, col);

            for (int row = 1; row < probabilities.rowCount(); row++) {
                if (probabilities.valueAt(row, col) > maxValue) {
                    maxValue = probabilities.valueAt(row, col);
                    maxIndex = row;
                }
            }

            predictions[col] = maxIndex;
        }

        return predictions;
    }

}
