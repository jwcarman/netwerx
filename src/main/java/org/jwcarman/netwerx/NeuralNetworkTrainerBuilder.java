package org.jwcarman.netwerx;

import org.jwcarman.netwerx.autoencoder.AutoencoderTrainer;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierTrainer;
import org.jwcarman.netwerx.classification.multi.MultiClassifierTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.layer.dense.DenseLayerConfig;
import org.jwcarman.netwerx.layer.dropout.DropoutLayerConfig;
import org.jwcarman.netwerx.listener.TrainingListener;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.regression.RegressionModelTrainer;
import org.jwcarman.netwerx.score.ScoringFunction;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;

import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * A builder interface for creating neural network trainers.
 *
 * @param <M> the type of matrix used in the neural network
 */
public interface NeuralNetworkTrainerBuilder<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Builds a new {@link NeuralNetworkTrainer} instance.
     *
     * @return a new {@link NeuralNetworkTrainer} instance
     */
    NeuralNetworkTrainer<M> build();

    /**
     * Creates a new {@link AutoencoderTrainer} instance.
     *
     * @return a new {@link AutoencoderTrainer} instance
     */
    AutoencoderTrainer<M> buildAutoencoderTrainer();

    /**
     * Creates a new {@link BinaryClassifierTrainer} instance.
     *
     * @return a new {@link BinaryClassifierTrainer} instance
     */
    BinaryClassifierTrainer<M> buildBinaryClassifierTrainer();

    /**
     * Creates a new {@link MultiClassifierTrainer} instance.
     *
     * @param outputClasses the number of output classes for the multi-class classification
     * @return a new {@link MultiClassifierTrainer} instance
     */
    MultiClassifierTrainer<M> buildMultiClassifierTrainer(int outputClasses);

    /**
     * Creates a new {@link RegressionModelTrainer} instance.
     *
     * @return a new {@link RegressionModelTrainer} instance
     */
    RegressionModelTrainer<M> buildRegressionModelTrainer();

    /**
     * Sets the default optimizer supplier for the neural network trainer.
     *
     * @param defaultOptimizerSupplier a supplier that provides the default optimizer
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> defaultOptimizer(Supplier<Optimizer<M>> defaultOptimizerSupplier);

    /**
     * Adds a dense layer to the neural network trainer using the default configuration.
     *
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> denseLayer();

    /**
     * Adds a dense layer to the neural network trainer with a custom configuration.
     *
     * @param configurer a consumer that configures the dense layer
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> denseLayer(Consumer<DenseLayerConfig<M>> configurer);

    /**
     * Adds a dropout layer to the neural network trainer using the default configuration.
     *
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> dropoutLayer();

    /**
     * Adds a dropout layer to the neural network trainer with a custom configuration.
     *
     * @param configurer a consumer that configures the dropout layer
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> dropoutLayer(Consumer<DropoutLayerConfig<M>> configurer);

    /**
     * Sets the training listener to the neural network trainer.
     *
     * @param listener the training listener to use
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> listener(TrainingListener listener);

    /**
     * Sets the loss function to the neural network trainer.
     *
     * @param lossFunction the loss function to use
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> lossFunction(LossFunction lossFunction);

    /**
     * Sets the scoring function to the neural network trainer.
     *
     * @param scoringFunction the scoring function to use
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> scoringFunction(ScoringFunction scoringFunction);

    /**
     * Sets the stopping advisor to the neural network trainer.
     *
     * @param stoppingAdvisor the stopping advisor to use
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> stoppingAdvisor(StoppingAdvisor stoppingAdvisor);

    /**
     * Sets the validation dataset to the neural network trainer.
     *
     * @param validationDataset the validation dataset to use
     * @return the current instance of the builder
     */
    NeuralNetworkTrainerBuilder<M> validationDataset(Dataset<M> validationDataset);

}
