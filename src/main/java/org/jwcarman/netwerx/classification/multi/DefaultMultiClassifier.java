package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public class DefaultMultiClassifier<M extends Matrix<M>> implements MultiClassifier<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork<M> network;
    private final Loss loss;
    private final int outputClasses;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultMultiClassifier(NeuralNetwork<M> network, Loss loss, int outputClasses) {
        this.network = network;
        this.loss = loss;
        this.outputClasses = outputClasses;
    }

// ------------------------ INTERFACE METHODS ------------------------

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

    @Override
    public void train(M x, int[] labels, OptimizerProvider<M> optimizerProvider, TrainingObserver observer) {
        network.train(x, convertLabels(x, labels), loss, optimizerProvider, observer);
    }

    private M convertLabels(M input, int[] labels) {
        return input.likeKind(outputClasses, labels.length)
                .map((row, col, value) -> labels[col] == row ? 1.0 : 0.0);
    }
}