package org.jwcarman.netwerx.layer.dropout;

import org.jwcarman.netwerx.layer.Layer;
import org.jwcarman.netwerx.layer.LayerBackprop;
import org.jwcarman.netwerx.layer.LayerBackpropResult;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.Random;

public class DropoutLayerTrainer<M extends Matrix<M>> implements LayerTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private final int inputSize;
    private final double dropoutRate;
    private final Random random;
    private final double scale;

// --------------------------- CONSTRUCTORS ---------------------------

    public DropoutLayerTrainer(int inputSize, double dropoutRate, Random random) {
        this.inputSize = inputSize;
        this.dropoutRate = dropoutRate;
        this.random = random;
        this.scale = 1.0 / (1.0 - dropoutRate);
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LayerTrainer ---------------------

    @Override
    public void applyUpdates(LayerUpdate<M> gradients) {
        // Do nothing as dropout does not have weights to update
    }

    @Override
    public int inputSize() {
        return inputSize;
    }

    @Override
    public int outputSize() {
        return inputSize;
    }

    @Override
    public Layer<M> createLayer() {
        throw new UnsupportedOperationException("Dropout layers are non-inference layers.");
    }

    @Override
    public LayerBackprop<M> forwardPass(M input) {
        M mask = input.map((_, _, _) -> scale * random.nextDouble() >= dropoutRate ? 1.0 : 0.0);
        M activations = input.elementMultiply(mask);
        return new DropoutLayerBackprop(activations, mask);
    }

    @Override
    public double regularizationPenalty() {
        return 0;
    }

    @Override
    public boolean isInference() {
        return false;
    }

// -------------------------- INNER CLASSES --------------------------

    private class DropoutLayerBackprop implements LayerBackprop<M> {

// ------------------------------ FIELDS ------------------------------

        private final M activations;
        private final M mask;

// --------------------------- CONSTRUCTORS ---------------------------

        public DropoutLayerBackprop(M activations, M mask) {
            this.activations = activations;
            this.mask = mask;
        }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LayerBackprop ---------------------

        @Override
        public M activations() {
            return activations;
        }

        @Override
        public LayerBackpropResult<M> apply(M outputGradient) {
            return new LayerBackpropResult<>(outputGradient.elementMultiply(mask), new LayerUpdate<>());
        }

    }

}
