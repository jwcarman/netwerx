package org.jwcarman.netwerx.autoencoder;

import org.jwcarman.netwerx.matrix.Matrix;

public interface AutoencoderTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Trains the autoencoder with the provided input data.
     *
     * @param input the input matrix, where each column represents a sample
     * @return the trained autoencoder instance
     */
    Autoencoder<M> train(M input);

}
