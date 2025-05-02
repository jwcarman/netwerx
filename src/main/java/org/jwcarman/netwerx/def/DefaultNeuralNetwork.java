package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.OptimizerProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

class DefaultNeuralNetwork<M extends Matrix<M>> implements NeuralNetwork<M> {

// ------------------------------ FIELDS ------------------------------

    private final List<Layer<M>> layers;

// --------------------------- CONSTRUCTORS ---------------------------

    DefaultNeuralNetwork(List<Layer<M>> layers) {
        this.layers = layers;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetwork ---------------------

    /**
     * Predicts the output for the given input using the neural network.
     *
     * @param x the input features as a matrix (each column is a feature vector)
     * @return the predicted output as a matrix (each column is a predicted output vector)
     */
    @Override
    public M predict(M x) {
        return layers.stream().reduce(x, (M acc, Layer<M> layer) -> layer.inference(acc), (a, b) -> a);
    }

    /**
     * Trains the neural network using the provided input and target output.
     *
     * @param x            the input features as a matrix (each column is a feature vector)
     * @param y            the target output as a matrix (each column is a target output vector)
     * @param lossFunction the loss function to use for training
     * @param observer     an observer to monitor training progress
     */
    @Override
    public void train(M x, M y, Loss lossFunction, OptimizerProvider<M> optimizerProvider, TrainingObserver observer) {
        int epoch = 1;
        boolean continueTraining;
        var weightOptimizers = IntStream.range(0, layers.size()).mapToObj(optimizerProvider::weightOptimizer).toList();
        var biasOptimizers = IntStream.range(0, layers.size()).mapToObj(optimizerProvider::biasOptimizer).toList();

        do {
            continueTraining = trainOneEpoch(x, y, lossFunction, observer, weightOptimizers, biasOptimizers, epoch);
            epoch++;
        } while (continueTraining);
    }

// -------------------------- OTHER METHODS --------------------------

    private boolean trainOneEpoch(M x, M y, Loss lossFunction, TrainingObserver observer, List<Optimizer<M>> weightOptimizers, List<Optimizer<M>> biasOptimizers, int epoch) {
        boolean continueTraining;
        final List<Backprop<M>> backProps = new ArrayList<>();
        M a = x;
        for (int i = 0; i < layers.size(); i++) {
            var layer = layers.get(i);
            var bp = layer.forward(a, weightOptimizers.get(i), biasOptimizers.get(i));
            backProps.addFirst(bp);
            a = bp.a();
        }
        var loss = lossFunction.loss(a, y);
        var gradOutput = lossFunction.gradient(a, y);
        for (Backprop<M> backProp : backProps) {
            gradOutput = backProp.apply(gradOutput);
        }
        continueTraining = observer.onEpoch(epoch, loss, a, y);
        return continueTraining;
    }

}
