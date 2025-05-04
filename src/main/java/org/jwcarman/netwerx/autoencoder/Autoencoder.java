package org.jwcarman.netwerx.autoencoder;

import org.jwcarman.netwerx.matrix.Matrix;

public interface Autoencoder<M extends Matrix<M>> {
    M encode(M input);
    M decode(M encoded);
    M reconstruct(M input);
}
