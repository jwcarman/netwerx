package org.jwcarman.netwerx.layer;

import org.jwcarman.netwerx.matrix.Matrix;

public interface LayerBackprop<M extends Matrix<M>> {
    M activations();
    LayerBackpropResult<M> apply(M outputGradient);
}
