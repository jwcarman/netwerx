package org.jwcarman.netwerx.layer.dense;

import org.jwcarman.netwerx.activation.ActivationFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.parameter.ParameterInitializer;
import org.jwcarman.netwerx.regularization.RegularizationFunction;

import java.util.function.Supplier;

public interface DenseLayerConfig<M extends Matrix<M>> {

    DenseLayerConfig<M> units(int units);
    DenseLayerConfig<M> activationFunction(ActivationFunction activationFunction);
    DenseLayerConfig<M> optimizers(Supplier<Optimizer<M>> optimizersSupplier);
    DenseLayerConfig<M> weightOptimizer(Supplier<Optimizer<M>> weightOptimizerSupplier);
    DenseLayerConfig<M> biasOptimizer(Supplier<Optimizer<M>> biasOptimizerSupplier);
    DenseLayerConfig<M> regularizationFunction(RegularizationFunction<M> regularizationFunction);
    DenseLayerConfig<M> weightInitializer(ParameterInitializer weightInitializer);
    DenseLayerConfig<M> biasInitializer(ParameterInitializer biasInitializer);
}
