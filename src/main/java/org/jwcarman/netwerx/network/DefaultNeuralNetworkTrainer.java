package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.batch.TrainingResult;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.layer.LayerBackprop;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.normalization.InputNormalizer;
import org.jwcarman.netwerx.normalization.NormalizationFunctionFactory;
import org.jwcarman.netwerx.util.Streams;
import org.jwcarman.netwerx.util.async.Tasks;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

public class DefaultNeuralNetworkTrainer<M extends Matrix<M>> implements NeuralNetworkTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private static final Logger LOGGER = LoggerFactory.getLogger(DefaultNeuralNetworkTrainer.class);

    private final List<LayerTrainer<M>> layerTrainers;
    private final NeuralNetworkTrainerConfig<M> config;
    private final NormalizationFunctionFactory defaultNormalizationFactory;
    private final Map<Integer, NormalizationFunctionFactory> normalizationFactories;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultNeuralNetworkTrainer(NeuralNetworkTrainerConfig<M> config, List<LayerTrainer<M>> layerTrainers, NormalizationFunctionFactory defaultNormalizationFactory, Map<Integer, NormalizationFunctionFactory> normalizationFactories) {
        this.layerTrainers = layerTrainers;
        this.config = config;
        this.defaultNormalizationFactory = defaultNormalizationFactory;
        this.normalizationFactories = normalizationFactories;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetworkTrainer ---------------------

    @Override
    public NeuralNetwork<M> train(Dataset<M> trainingDataset) {
        if (layerTrainers.getFirst().inputSize() != trainingDataset.features().rowCount()) {
            throw new IllegalArgumentException(String.format("Dataset input must have input size %d.", layerTrainers.getFirst().inputSize()));
        }

        final var normalizer = InputNormalizer.forDataset(defaultNormalizationFactory, normalizationFactories, trainingDataset);

        final var normalizedDatasets = normalize(trainingDataset, normalizer);

        return runTrainingLoop(normalizedDatasets, () -> createNeuralNetwork(normalizer));
    }

// -------------------------- OTHER METHODS --------------------------

    private DefaultNeuralNetwork<M> createNeuralNetwork(UnaryOperator<M> normalizer) {
        return new DefaultNeuralNetwork<>(normalizer, layerTrainers.stream()
                .filter(LayerTrainer::isInference)
                .map(LayerTrainer::createLayer)
                .toList());
    }

    private NormalizedDatasets<M> normalize(Dataset<M> training, UnaryOperator<M> normalizer) {
        return new NormalizedDatasets<>(
                new Dataset<>(normalizer.apply(training.features()), training.labels()),
                new Dataset<>(normalizer.apply(config.validationDataset().features()), config.validationDataset().labels())
        );
    }

    private NeuralNetwork<M> runTrainingLoop(
            NormalizedDatasets<M> data,
            Supplier<NeuralNetwork<M>> snapshotter) {
        var bestScore = Double.NEGATIVE_INFINITY;
        var bestNetwork = snapshotter.get();
        var bestEpoch = -1;
        var epoch = 1;
        LOGGER.info("Beginning training with {} samples using batch size {} and sub-batch count {}...",data.training().size(), config.batchSize(), config.subBatchCount());
        final var before = System.nanoTime();
        while (true) {
            var score = trainEpoch(epoch, data);
            if (score > bestScore) {
                bestScore = score;
                bestEpoch = epoch;
                bestNetwork = snapshotter.get();
            }
            if (config.stoppingAdvisor().shouldStop(epoch, score)) {
                break;
            }
            epoch++;
        }
        final var after = System.nanoTime();
        LOGGER.info("Finished training in {} seconds after {} epochs with best score {} at epoch {}.", Duration.ofNanos(after - before).toMillis() / 1000.0, epoch, bestScore, bestEpoch);
        return bestNetwork;
    }

    private double trainEpoch(int epoch, NormalizedDatasets<M> datasets) {
        var batches = datasets.training().shuffle(config.random()).batches(config.batchSize());
        var totalLoss = 0.0;
        for (Dataset<M> batch : batches) {
            var result = trainBatch(batch);
            totalLoss += (result.trainingLoss() * batch.size());
            applyLayerUpdates(epoch, result.layerUpdates());
        }
        var trainingLoss = totalLoss / datasets.training.size();
        var regularizationPenalty = calculateRegularizationPenalty();
        var validationLoss = calculateValidationLoss(datasets.validation());
        var outcome = new EpochOutcome(epoch, trainingLoss, validationLoss, regularizationPenalty, trainingLoss + regularizationPenalty);
        config.listener().onEpoch(outcome);
        return config.scoringFunction().score(outcome);
    }

    private TrainingResult<M> trainBatch(Dataset<M> batch) {
        var subBatches = batch.partition(config.subBatchCount());

        var results = Tasks.executeAll(subBatches.stream()
                .map(subBatch -> (Supplier<TrainingResult<M>>) () -> trainSubBatch(subBatch))
                .toList());
        return TrainingResult.aggregate(results);
    }

    private void applyLayerUpdates(int epoch, List<LayerUpdate<M>> layerUpdates) {
        Streams.zip(layerTrainers.stream(), layerUpdates.stream())
                .forEach(pair -> pair.left().applyUpdates(epoch, pair.right()));
    }

    private double calculateRegularizationPenalty() {
        return layerTrainers.stream()
                .mapToDouble(LayerTrainer::regularizationPenalty)
                .sum();
    }

    private double calculateValidationLoss(Dataset<M> validationSet) {
        if (validationSet.features().isEmpty()) {
            return Double.NaN;
        }
        var inferred = layerTrainers.stream().reduce(validationSet.features(), (M acc, LayerTrainer<M> layer) -> layer.forwardPass(acc).activations(), (a, _) -> a);
        return config.lossFunction().loss(inferred, validationSet.labels());
    }

    private TrainingResult<M> trainSubBatch(Dataset<M> subBatch) {
        var forwardPassResult = performForwardPass(subBatch);
        var trainingLoss = config.lossFunction().loss(forwardPassResult.output(), subBatch.labels());
        var outputGradient = config.lossFunction().gradient(forwardPassResult.output(), subBatch.labels());
        var layerUpdates = new ArrayList<LayerUpdate<M>>();

        for (LayerBackprop<M> backProp : forwardPassResult.backProps()) {
            var result = backProp.apply(outputGradient);
            layerUpdates.addFirst(result.layerUpdate());
            outputGradient = result.outputGradient();
        }
        return new TrainingResult<>(subBatch.size(), trainingLoss, layerUpdates);
    }

    private ForwardPassResult<M> performForwardPass(Dataset<M> trainingDataset) {
        M activations = trainingDataset.features();
        var backProps = new ArrayList<LayerBackprop<M>>();
        for (LayerTrainer<M> trainer : layerTrainers) {
            var bp = trainer.forwardPass(activations);
            backProps.addFirst(bp);
            activations = bp.activations();
        }
        return new ForwardPassResult<>(activations, backProps);
    }

// -------------------------- INNER CLASSES --------------------------

    private record ForwardPassResult<M extends Matrix<M>>(M output, ArrayList<LayerBackprop<M>> backProps) {

    }

    private record NormalizedDatasets<M extends Matrix<M>>(Dataset<M> training, Dataset<M> validation) {

    }

}
