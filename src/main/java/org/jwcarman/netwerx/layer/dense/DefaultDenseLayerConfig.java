package org.jwcarman.netwerx.layer.dense;

import org.jwcarman.netwerx.activation.ActivationFunction;
import org.jwcarman.netwerx.activation.ActivationFunctions;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.parameter.ParameterInitializer;
import org.jwcarman.netwerx.parameter.ParameterInitializers;
import org.jwcarman.netwerx.regularization.RegularizationFunction;
import org.jwcarman.netwerx.regularization.Regularizations;

import java.util.function.Supplier;

public class DefaultDenseLayerConfig<M extends Matrix<M>> implements DenseLayerConfig<M> {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_INITIAL_BIAS = 0.01;
    private final int inputSize;
    private int units = 8;
    private ActivationFunction activationFunction = ActivationFunctions.relu();
    private Supplier<Optimizer<M>> weightOptimizerSupplier = Optimizers::sgd;
    private Supplier<Optimizer<M>> biasOptimizerSupplier = Optimizers::sgd;
    private RegularizationFunction<M> regularizationFunction = Regularizations.noop();
    private ParameterInitializer weightInitializer = ParameterInitializers.heUniform();
    private ParameterInitializer biasInitializer = ParameterInitializers.constant(DEFAULT_INITIAL_BIAS);

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultDenseLayerConfig(int inputSize) {
        this.inputSize = inputSize;
    }

// --------------------- GETTER / SETTER METHODS ---------------------

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public ParameterInitializer getBiasInitializer() {
        return biasInitializer;
    }

    public Supplier<Optimizer<M>> getBiasOptimizerSupplier() {
        return biasOptimizerSupplier;
    }

    public int getInputSize() {
        return inputSize;
    }

    public RegularizationFunction<M> getRegularizationFunction() {
        return regularizationFunction;
    }

    public int getUnits() {
        return units;
    }

    public ParameterInitializer getWeightInitializer() {
        return weightInitializer;
    }

    public Supplier<Optimizer<M>> getWeightOptimizerSupplier() {
        return weightOptimizerSupplier;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface DenseLayerConfig ---------------------


    @Override
    public DenseLayerConfig<M> weightInitializer(ParameterInitializer weightInitializer) {
        this.weightInitializer = weightInitializer;
        return this;
    }

    @Override
    public DenseLayerConfig<M> biasInitializer(ParameterInitializer biasInitializer) {
        this.biasInitializer = biasInitializer;
        return this;
    }

    @Override
    public DenseLayerConfig<M> units(int units) {
        this.units = units;
        return this;
    }

    @Override
    public DenseLayerConfig<M> activationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    @Override
    public DenseLayerConfig<M> optimizers(Supplier<Optimizer<M>> optimizersSupplier) {
        this.weightOptimizerSupplier = optimizersSupplier;
        this.biasOptimizerSupplier = optimizersSupplier;
        return this;
    }

    @Override
    public DenseLayerConfig<M> weightOptimizer(Supplier<Optimizer<M>> weightOptimizerSupplier) {
        this.weightOptimizerSupplier = weightOptimizerSupplier;
        return this;
    }

    @Override
    public DenseLayerConfig<M> biasOptimizer(Supplier<Optimizer<M>> biasOptimizerSupplier) {
        this.biasOptimizerSupplier = biasOptimizerSupplier;
        return this;
    }

    public DenseLayerConfig<M> regularizationFunction(RegularizationFunction<M> regularizationFunction) {
        this.regularizationFunction = regularizationFunction;
        return this;
    }

}
