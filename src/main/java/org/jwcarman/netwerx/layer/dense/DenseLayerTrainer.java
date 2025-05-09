package org.jwcarman.netwerx.layer.dense;

import org.jwcarman.netwerx.activation.ActivationFunction;
import org.jwcarman.netwerx.layer.Layer;
import org.jwcarman.netwerx.layer.LayerBackprop;
import org.jwcarman.netwerx.layer.LayerBackpropResult;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.regularization.RegularizationFunction;

import java.util.Random;

public class DenseLayerTrainer<M extends Matrix<M>> implements LayerTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    public static final String WEIGHTS_GRADIENT = "dW";
    public static final String BIASES_GRADIENT = "db";
    private M weights;
    private M biases;
    private final ActivationFunction activationFunction;
    private final Optimizer<M> weightsOptimizer;
    private final Optimizer<M> biasesOptimizer;
    private final RegularizationFunction<M> regularizationFunction;

// --------------------------- CONSTRUCTORS ---------------------------

    public DenseLayerTrainer(MatrixFactory<M> factory, Random random, DefaultDenseLayerConfig<M> config) {
        this.activationFunction = config.getActivationFunction();
        this.weights = factory.filled(config.getUnits(), config.getInputSize(), () -> config.getWeightInitializer().initialize(random, config.getInputSize(), config.getUnits()));
        this.biases = factory.filled(config.getUnits(), 1, () -> config.getBiasInitializer().initialize(random, config.getInputSize(), config.getUnits()));
        this.weightsOptimizer = config.getWeightOptimizerSupplier().get();
        this.biasesOptimizer = config.getBiasOptimizerSupplier().get();
        this.regularizationFunction = config.getRegularizationFunction();
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LayerTrainer ---------------------

    @Override
    public int inputSize() {
        return weights.columnCount();
    }

    @Override
    public int outputSize() {
        return biases.rowCount();
    }

    @Override
    public LayerBackprop<M> forwardPass(M aPrev) {
        final var z = weights.multiply(aPrev).addColumnVector(biases);
        final var a = activationFunction.apply(z);
        return new Backprop(aPrev, z, a);
    }

    @Override
    public void applyUpdates(LayerUpdate<M> gradients) {
        weights = weightsOptimizer.optimize(weights, gradients.gradient(WEIGHTS_GRADIENT));
        biases = biasesOptimizer.optimize(biases, gradients.gradient(BIASES_GRADIENT));
    }

    @Override
    public Layer<M> createLayer() {
        return new DenseLayer<>(weights, biases, activationFunction);
    }

    @Override
    public double regularizationPenalty() {
        return regularizationFunction.penalty(weights);
    }

// -------------------------- INNER CLASSES --------------------------

    private class Backprop implements LayerBackprop<M> {

// ------------------------------ FIELDS ------------------------------

        private final M aPrev;
        private final M z;
        private final M a;

// --------------------------- CONSTRUCTORS ---------------------------

        public Backprop(M aPrev, M z, M a) {
            this.aPrev = aPrev;
            this.z = z;
            this.a = a;
        }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LayerBackprop ---------------------

        @Override
        public M activations() {
            return a;
        }

        @Override
        public LayerBackpropResult<M> apply(M outputGradient) {
            var dz = outputGradient.elementMultiply(activationFunction.derivative(z));
            var m = outputGradient.columnCount();

            var dW = dz.multiply(aPrev.transpose()).elementDivide(m);
            var dRegularization = regularizationFunction.gradient(weights);
            var db = dz.rowSum().elementDivide(m);

            var gradients = new LayerUpdate<M>();
            gradients.addGradient(WEIGHTS_GRADIENT, dW.add(dRegularization));
            gradients.addGradient(BIASES_GRADIENT, db);

            return new LayerBackpropResult<>(weights.transpose().multiply(dz), gradients);
        }

    }

}
