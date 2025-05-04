package org.jwcarman.netwerx;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.observer.TrainingObservers;

public interface NeuralNetworkTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    default NeuralNetwork<M> train(Dataset<M> trainingDataset) {
        return train(trainingDataset, TrainingObservers.noop());
    }

    NeuralNetwork<M> train(Dataset<M> trainingDataset, TrainingObserver observer);

}
