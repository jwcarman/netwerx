package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

public class DefaultBinaryClassifier<M extends Matrix<M>> implements BinaryClassifier<M> {

// ------------------------------ FIELDS ------------------------------

    public static final double TRUE = 1.0;
    public static final double FALSE = 0.0;
    public static final double THRESHOLD = 0.5;
    private final NeuralNetwork<M> network;
    private final Loss loss;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultBinaryClassifier(NeuralNetwork<M> network, Loss loss) {
        this.network = network;
        this.loss = loss;
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

    @Override
    public void train(M x, boolean[] labels, OptimizerProvider<M> optimizerProvider, TrainingObserver observer) {
        network.train(x, convertLabels(x, labels), loss, optimizerProvider, observer);
    }

    private M convertLabels(M input, boolean[] labels) {
        return input.likeKind(1, input.columnCount()).map((row, col, value) -> labels[col] ? TRUE : FALSE);
    }
}
