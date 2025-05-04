package org.jwcarman.netwerx.autoencoder;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.matrix.Matrix;

public class DefaultAutoencoder<M extends Matrix<M>> implements Autoencoder<M> {

// ------------------------------ FIELDS ------------------------------

    private final NeuralNetwork<M> encoder;
    private final NeuralNetwork<M> decoder;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultAutoencoder(NeuralNetwork<M> encoder, NeuralNetwork<M> decoder) {
        this.encoder = encoder;
        this.decoder = decoder;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Autoencoder ---------------------

    @Override
    public M encode(M input) {
        return encoder.predict(input);
    }

    @Override
    public M decode(M encoded) {
        return decoder.predict(encoded);
    }

    @Override
    public M reconstruct(M input) {
        return decode(encode(input));
    }
}
