package org.jwcarman.netwerx.autoencoder;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.observer.TrainingObservers;

public interface AutoencoderTrainer<M extends Matrix<M>> {

    /**
     * Trains the autoencoder with the provided input data.
     *
     * @param input the input matrix, where each column represents a sample
     * @return the trained autoencoder instance
     */
    default Autoencoder<M> train(M input) {
        return train(input, TrainingObservers.noop());
    }

    /**
     * Trains the autoencoder with the provided input data and a training observer.
     *
     * @param input    the input matrix, where each column represents a sample
     * @param observer the training observer to monitor the training process
     * @return the trained autoencoder instance
     */
    Autoencoder<M> train(M input, TrainingObserver observer);
}
