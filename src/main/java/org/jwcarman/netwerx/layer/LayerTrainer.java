package org.jwcarman.netwerx.layer;

import org.jwcarman.netwerx.matrix.Matrix;

public interface LayerTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    void applyUpdates(LayerUpdate<M> gradients);

    int inputSize();

    int outputSize();

    Layer<M> createLayer();

    LayerBackprop<M> forwardPass(M input);

    double regularizationPenalty();

    default boolean isInference() {
        return true;
    }
}
