package org.jwcarman.netwerx.def;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;

import java.util.ArrayList;
import java.util.List;

class DefaultNeuralNetwork implements NeuralNetwork {

// ------------------------------ FIELDS ------------------------------

    private final List<Layer> layers;

// --------------------------- CONSTRUCTORS ---------------------------

    DefaultNeuralNetwork(List<Layer> layers) {
        this.layers = layers;
    }

// -------------------------- OTHER METHODS --------------------------

    /**
     * Predicts the output for the given input using the neural network.
     *
     * @param x the input features as a SimpleMatrix (each column is a feature vector)
     * @return the predicted output as a SimpleMatrix (each column is a predicted output vector)
     */
    @Override
    public SimpleMatrix predict(SimpleMatrix x) {
        return layers.stream().reduce(x,
                (currentInput, layer) -> layer.forward(currentInput).a(),
                (a, _) -> a);
    }

    /**
     * Trains the neural network using the provided input and target output.
     *
     * @param x            the input features as a SimpleMatrix (each column is a feature vector)
     * @param y            the target output as a SimpleMatrix (each column is a target output vector)
     * @param lossFunction the loss function to use for training
     * @param observer     an observer to monitor training progress
     */
    @Override
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
