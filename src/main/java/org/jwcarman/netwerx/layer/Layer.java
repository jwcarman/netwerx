package org.jwcarman.netwerx.layer;

import org.jwcarman.netwerx.matrix.Matrix;

public interface Layer<M extends Matrix<M>> {
    M apply(M input);
}
