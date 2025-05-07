package org.jwcarman.netwerx;

import org.jwcarman.netwerx.autoencoder.AutoencoderTrainer;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierTrainer;
import org.jwcarman.netwerx.classification.multi.MultiClassifierTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.regression.RegressionModelTrainer;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;

import java.util.function.Consumer;
import java.util.function.Supplier;

public interface NeuralNetworkTrainerBuilder<M extends Matrix<M>> {
    NeuralNetworkTrainerBuilder<M> denseLayer();
    NeuralNetworkTrainerBuilder<M> denseLayer(Consumer<DenseLayerConfig<M>> configurer);
    NeuralNetworkTrainerBuilder<M> dropoutLayer();
    NeuralNetworkTrainerBuilder<M> dropoutLayer(Consumer<DropoutLayerConfig<M>> configurer);
    NeuralNetworkTrainerBuilder<M> defaultOptimizer(Supplier<Optimizer<M>> defaultOptimizerSupplier);
    NeuralNetworkTrainerBuilder<M> stoppingAdvisor(StoppingAdvisor stoppingAdvisor);
    NeuralNetworkTrainerBuilder<M> lossFunction(LossFunction lossFunction);
    NeuralNetworkTrainerBuilder<M> validationDataset(Dataset<M> validationDataset);
    NeuralNetworkTrainer<M> build();

    RegressionModelTrainer<M> buildRegressionModelTrainer();

    BinaryClassifierTrainer<M> buildBinaryClassifierTrainer();

    MultiClassifierTrainer<M> buildMultiClassifierTrainer(int outputClasses);

    AutoencoderTrainer<M> buildAutoencoderTrainer();
}
