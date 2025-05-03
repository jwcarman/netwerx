package org.jwcarman.netwerx;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public interface NeuralNetworkTrainer<M extends Matrix<M>> {
    NeuralNetwork<M> train(Dataset<M> trainingDataset, TrainingObserver observer);
}
