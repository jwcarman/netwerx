package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.loss.LossFunction;

public class NeuralNetworkBinaryClassifier implements BinaryClassifier {

// ------------------------------ FIELDS ------------------------------

    public static final double TRUE = 1.0;
    public static final double FALSE = 0.0;
    public static final double THRESHOLD = 0.5;
    private final NeuralNetwork network;
    private final LossFunction lossFunction;

// --------------------------- CONSTRUCTORS ---------------------------

    NeuralNetworkBinaryClassifier(NeuralNetwork network, LossFunction lossFunction) {
        this.network = network;
        this.lossFunction = lossFunction;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassifier ---------------------

    @Override
    public SimpleMatrix predict(SimpleMatrix input) {
        var probabilities = network.predict(input);
        return probabilities.elementOp((_, _, value) -> value >= THRESHOLD ? TRUE : FALSE);
    }

    @Override
    public void train(SimpleMatrix x, SimpleMatrix y, TrainingObserver observer) {
        network.train(x, y, lossFunction, observer);
    }

}
