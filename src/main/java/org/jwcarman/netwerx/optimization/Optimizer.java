package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.matrix.Matrix;

@FunctionalInterface
public interface Optimizer<M extends Matrix<M>> {
    M optimize(M parameter, M gradient);
}
