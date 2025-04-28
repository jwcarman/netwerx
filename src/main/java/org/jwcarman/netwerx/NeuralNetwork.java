package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.loss.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

// ------------------------------ FIELDS ------------------------------

    private final List<Layer> layers;

// --------------------------- CONSTRUCTORS ---------------------------

    NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
    }

// -------------------------- OTHER METHODS --------------------------

    public SimpleMatrix predict(SimpleMatrix x) {
        return layers.stream().reduce(x,
                (currentInput, layer) -> layer.forward(currentInput).a(),
                (a, b) -> a);
    }

    public SimpleMatrix predict(double[] features) {
        return predict(new SimpleMatrix(features.length, 1, true, features));
    }

    public void train(SimpleMatrix x, SimpleMatrix y, LossFunction lossFunction, TrainingObserver observer) {
        int epoch = 1;
        boolean continueTraining;
        do {
            var a = x;
            var backProps = new ArrayList<Backprop>();
            for (Layer layer : layers) {
                var backpropagator = layer.forward(a);
                backProps.addFirst(backpropagator);
                a = backpropagator.a();
            }
            var loss = lossFunction.loss(a, y);
            var gradOutput = lossFunction.gradient(a, y);
            for (Backprop backProp : backProps) {
                gradOutput = backProp.apply(gradOutput);
            }
            continueTraining = observer.onEpoch(epoch, loss, a, y);
            epoch++;
        } while(continueTraining);
    }

}
