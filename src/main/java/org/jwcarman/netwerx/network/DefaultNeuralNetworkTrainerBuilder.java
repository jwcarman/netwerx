package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.layer.dense.DenseLayerConfig;
import org.jwcarman.netwerx.layer.dropout.DropoutLayerConfig;
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
import org.jwcarman.netwerx.listener.TrainingListener;
import org.jwcarman.netwerx.listener.TrainingListeners;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.loss.LossFunctions;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.parameter.ParameterInitializers;
import org.jwcarman.netwerx.regression.DefaultRegressionModelTrainer;
import org.jwcarman.netwerx.regression.RegressionModelTrainer;
import org.jwcarman.netwerx.score.ScoringFunction;
import org.jwcarman.netwerx.score.ScoringFunctions;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;
import org.jwcarman.netwerx.stopping.StoppingAdvisors;
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
    private StoppingAdvisor stoppingAdvisor = StoppingAdvisors.patience();
    private LossFunction lossFunction = LossFunctions.mse();
    private Dataset<M> validationDataset;
    private TrainingExecutor<M> trainingExecutor = new FullBatchExecutor<>();
    private ScoringFunction scoringFunction = ScoringFunctions.validationLossWithPenalty();
    private TrainingListener listener = TrainingListeners.noop();

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
    public NeuralNetworkTrainerBuilder<M> listener(TrainingListener listener) {
        this.listener = listener;
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
    public NeuralNetworkTrainerBuilder<M> scoringFunction(ScoringFunction scoringFunction) {
        this.scoringFunction = scoringFunction;
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
        var config = new NeuralNetworkTrainerConfig<>(lossFunction, validationDataset, trainingExecutor, scoringFunction, stoppingAdvisor, listener);
        return new DefaultNeuralNetworkTrainer<>(config, layerTrainers);
    }

    @Override
    public RegressionModelTrainer<M> buildRegressionModelTrainer() {
        lossFunction(LossFunctions.mse());
        denseLayer(layer -> layer
                .activationFunction(ActivationFunctions.linear())
                .weightInitializer(ParameterInitializers.xavierUniform())
                .biasInitializer(ParameterInitializers.zeros())
                .units(1));
        return new DefaultRegressionModelTrainer<>(build());
    }

    @Override
    public BinaryClassifierTrainer<M> buildBinaryClassifierTrainer() {
        lossFunction(LossFunctions.bce());
        denseLayer(layer -> layer
                .units(1)
                .activationFunction(ActivationFunctions.sigmoid())
                .weightInitializer(ParameterInitializers.xavierUniform())
                .biasInitializer(ParameterInitializers.zeros()));
        return new DefaultBinaryClassifierTrainer<>(build());
    }

    @Override
    public MultiClassifierTrainer<M> buildMultiClassifierTrainer(int outputClasses) {
        lossFunction(LossFunctions.cce());
        denseLayer(layer -> layer
                .units(outputClasses)
                .activationFunction(ActivationFunctions.softmax())
                .weightInitializer(ParameterInitializers.xavierUniform())
                .biasInitializer(ParameterInitializers.zeros()));
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
