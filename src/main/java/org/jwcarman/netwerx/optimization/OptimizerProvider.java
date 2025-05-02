package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.matrix.Matrix;

public interface OptimizerProvider<M extends Matrix<M>> {
    Optimizer<M> weightOptimizer(int layer);

    Optimizer<M> biasOptimizer(int layer);
}
