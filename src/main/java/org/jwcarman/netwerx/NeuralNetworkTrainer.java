package org.jwcarman.netwerx;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

public interface NeuralNetworkTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    NeuralNetwork<M> train(Dataset<M> trainingDataset);

}
