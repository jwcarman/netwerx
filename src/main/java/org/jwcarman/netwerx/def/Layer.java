package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.optimization.Optimizer;

import java.util.Random;

class Layer<M extends Matrix<M>> {

// ------------------------------ FIELDS ------------------------------

    private M weights;
    private M biases;
    private final Activation activation;


// --------------------------- CONSTRUCTORS ---------------------------

    public Layer(MatrixFactory<M> factory, Random random, DefaultLayerConfig config) {
        this.weights = factory.filled(config.getUnits(), config.getInputSize(), () -> config.getActivation().generateInitialWeight(random, config.getInputSize(), config.getUnits()));
        this.biases = factory.filled(config.getUnits(), 1, () -> config.getActivation().generateInitialBias());
        this.activation = config.getActivation();
    }

// -------------------------- OTHER METHODS --------------------------

    public Backprop<M> forward(final M aPrev, Optimizer<M> weightOptimizer, Optimizer<M> biasOptimizer) {
        final var z = weights.multiply(aPrev).addColumnVector(biases);
        final var a = activation.apply(z);
        return new LayerBackprop(aPrev, z, a, weightOptimizer, biasOptimizer);
    }

    public M inference(final M aPrev) {
        final var z = weights.multiply(aPrev).addColumnVector(biases);
        return activation.apply(z);
    }

// -------------------------- INNER CLASSES --------------------------

    private class LayerBackprop implements Backprop<M> {

// ------------------------------ FIELDS ------------------------------

        private final M aPrev;
        private final M z;
        private final M a;
        private final Optimizer<M> weightOptimizer;
        private final Optimizer<M> biasOptimizer;

// --------------------------- CONSTRUCTORS ---------------------------

        public LayerBackprop(M aPrev, M z, M a, Optimizer<M> weightOptimizer, Optimizer<M> biasOptimizer) {
            this.aPrev = aPrev;
            this.z = z;
            this.a = a;
            this.weightOptimizer = weightOptimizer;
            this.biasOptimizer = biasOptimizer;
        }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Backprop ---------------------

        @Override
        public M a() {
            return a;
        }

        @Override
        public M apply(M gradOutput) {
            final var dz = gradOutput.elementMultiply(activation.derivative(z));

            final var m = gradOutput.columnCount();

            final var originalWeights = weights.copy();

            final var dw = dz.multiply(aPrev.transpose()).elementDivide(m);
            weights = weightOptimizer.optimize(weights, dw);

            final var db = dz.rowSum().elementDivide(m);
            biases = biasOptimizer.optimize(biases, db);

            return originalWeights.transpose().multiply(dz);
        }

    }

}
