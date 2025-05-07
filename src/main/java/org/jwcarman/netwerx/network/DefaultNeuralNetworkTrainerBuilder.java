package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.DenseLayerConfig;
import org.jwcarman.netwerx.DropoutLayerConfig;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.NeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.activation.ActivationFunctions;
import org.jwcarman.netwerx.autoencoder.AutoencoderTrainer;
import org.jwcarman.netwerx.autoencoder.DefaultAutoencoderTrainer;
import org.jwcarman.netwerx.batch.FullBatchExecutor;
import org.jwcarman.netwerx.batch.TrainingExecutor;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierTrainer;
import org.jwcarman.netwerx.classification.binary.DefaultBinaryClassifierTrainer;
import org.jwcarman.netwerx.classification.multi.DefaultMultiClassifierTrainer;
import org.jwcarman.netwerx.classification.multi.MultiClassifierTrainer;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.layer.dense.DefaultDenseLayerConfig;
import org.jwcarman.netwerx.layer.dense.DenseLayerTrainer;
import org.jwcarman.netwerx.layer.dropout.DefaultDropoutLayerConfig;
import org.jwcarman.netwerx.layer.dropout.DropoutLayerTrainer;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regression.DefaultRegressionModelTrainer;
import org.jwcarman.netwerx.regression.RegressionModelTrainer;
import org.jwcarman.netwerx.stopping.EpochCountStoppingAdvisor;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;
import org.jwcarman.netwerx.util.Randoms;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class DefaultNeuralNetworkTrainerBuilder<M extends Matrix<M>> implements NeuralNetworkTrainerBuilder<M> {

// ------------------------------ FIELDS ------------------------------

    private final MatrixFactory<M> matrixFactory;
    private final Random random;
    private final List<LayerTrainer<M>> layerTrainers = new ArrayList<>();
    private int inputSize;
    private Supplier<Optimizer<M>> defaultOptimizerSupplier = Optimizers::sgd;
    private StoppingAdvisor stoppingAdvisor = new EpochCountStoppingAdvisor();
    private LossFunction lossFunction = Losses.mse();
    private Dataset<M> validationDataset;
    private TrainingExecutor<M> trainingExecutor = new FullBatchExecutor<>();

// -------------------------- STATIC METHODS --------------------------

    private static <T> Consumer<T> noopConsumer() {
        return _ -> {
        };
    }

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultNeuralNetworkTrainerBuilder(MatrixFactory<M> matrixFactory, int inputSize) {
        this(matrixFactory, inputSize, Randoms.defaultRandom());
    }

    public DefaultNeuralNetworkTrainerBuilder(MatrixFactory<M> matrixFactory, int inputSize, Random random) {
        this.matrixFactory = matrixFactory;
        this.inputSize = inputSize;
        this.random = random;
        this.validationDataset = new Dataset<>(matrixFactory.empty(), matrixFactory.empty());
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetworkTrainerBuilder ---------------------

    @Override
    public NeuralNetworkTrainerBuilder<M> denseLayer() {
        return denseLayer(noopConsumer());
    }

    @Override
    public NeuralNetworkTrainerBuilder<M> denseLayer(Consumer<DenseLayerConfig<M>> configurer) {
        var config = new DefaultDenseLayerConfig<M>(inputSize);
        config.optimizers(defaultOptimizerSupplier);
        configurer.accept(config);
        layerTrainers.add(new DenseLayerTrainer<>(matrixFactory, random, config));
        inputSize = config.getUnits();
        return this;
    }

    @Override
    public NeuralNetworkTrainerBuilder<M> dropoutLayer() {
        return dropoutLayer(noopConsumer());
    }

    @Override
    public NeuralNetworkTrainerBuilder<M> dropoutLayer(Consumer<DropoutLayerConfig<M>> configurer) {
        var config = new DefaultDropoutLayerConfig<M>();
        configurer.accept(config);
        layerTrainers.add(new DropoutLayerTrainer<>(inputSize, config.getDropoutRate(), random));
        return this;
    }

    public NeuralNetworkTrainerBuilder<M> defaultOptimizer(Supplier<Optimizer<M>> optimizerSupplier) {
        this.defaultOptimizerSupplier = optimizerSupplier;
        return this;
    }

    @Override
    public NeuralNetworkTrainerBuilder<M> stoppingAdvisor(StoppingAdvisor stoppingAdvisor) {
        this.stoppingAdvisor = stoppingAdvisor;
        return this;
    }

    @Override
    public NeuralNetworkTrainerBuilder<M> lossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        return this;
    }

    @Override
    public NeuralNetworkTrainerBuilder<M> validationDataset(Dataset<M> validationDataset) {
        this.validationDataset = validationDataset;
        return this;
    }

    @Override
    public NeuralNetworkTrainer<M> build() {
        return new DefaultNeuralNetworkTrainer<>(layerTrainers, stoppingAdvisor, lossFunction, validationDataset, trainingExecutor);
    }

    @Override
    public RegressionModelTrainer<M> buildRegressionModelTrainer() {
        lossFunction(Losses.mse());
        denseLayer(layer -> layer
                .activationFunction(ActivationFunctions.linear())
                .units(1));
        return new DefaultRegressionModelTrainer<>(build());
    }

    @Override
    public BinaryClassifierTrainer<M> buildBinaryClassifierTrainer() {
        lossFunction(Losses.bce());
        denseLayer(layer -> layer.units(1).activationFunction(ActivationFunctions.sigmoid()));
        return new DefaultBinaryClassifierTrainer<>(build());
    }

    @Override
    public MultiClassifierTrainer<M> buildMultiClassifierTrainer(int outputClasses) {
        lossFunction(Losses.cce());
        denseLayer(layer -> layer
                .units(outputClasses)
                .activationFunction(ActivationFunctions.softmax()));
        return new DefaultMultiClassifierTrainer<>(build(), outputClasses);
    }

    @Override
    public AutoencoderTrainer<M> buildAutoencoderTrainer() {
        if (layerTrainers.size() < 3) {
            throw new IllegalStateException("An autoencoder must have at least three layers: input, hidden, and output.");
        }
        if (layerTrainers.getFirst().inputSize() != layerTrainers.getLast().outputSize()) {
            throw new IllegalStateException("The first layer's input size must match the last layer's output size for an autoencoder.");
        }
        return new DefaultAutoencoderTrainer<>(build());
    }

// -------------------------- OTHER METHODS --------------------------

    public NeuralNetworkTrainerBuilder<M> trainingExecutor(TrainingExecutor<M> trainingExecutor) {
        this.trainingExecutor = trainingExecutor;
        return this;
    }

}
