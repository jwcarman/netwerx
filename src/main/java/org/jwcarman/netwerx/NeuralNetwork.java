package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.loss.Loss;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

// ------------------------------ FIELDS ------------------------------

    private final List<Layer> layers;

// --------------------------- CONSTRUCTORS ---------------------------

    NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
    }

    public static NeuralNetworkBuilder builder(int inputSize) {
        return new NeuralNetworkBuilder(inputSize);
    }

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the output for the given input using the neural network.
     *
     * @param x the input features as a SimpleMatrix (each column is a feature vector)
     * @return the predicted output as a SimpleMatrix (each column is a predicted output vector)
     */
    public SimpleMatrix predict(SimpleMatrix x) {
        return layers.stream().reduce(x,
                (currentInput, layer) -> layer.forward(currentInput).a(),
                (a, b) -> a);
    }

    /**
     * Trains the neural network using the provided input and target output.
     *
     * @param x            the input features as a SimpleMatrix (each column is a feature vector)
     * @param y            the target output as a SimpleMatrix (each column is a target output vector)
     * @param lossFunction the loss function to use for training
     * @param observer     an observer to monitor training progress
     */
    public void train(SimpleMatrix x, SimpleMatrix y, Loss lossFunction, TrainingObserver observer) {
        int epoch = 1;
        boolean continueTraining;
        do {
            var a = x;
            var backProps = new ArrayList<Layer.Backprop>();
            for (Layer layer : layers) {
                var backpropagator = layer.forward(a);
                backProps.addFirst(backpropagator);
                a = backpropagator.a();
            }
            var loss = lossFunction.loss(a, y);
            var gradOutput = lossFunction.gradient(a, y);
            for (Layer.Backprop backProp : backProps) {
                gradOutput = backProp.apply(gradOutput);
            }
            continueTraining = observer.onEpoch(epoch, loss, a, y);
            epoch++;
        } while (continueTraining);
    }

}
