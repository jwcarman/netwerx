package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.EpochOutcome;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.batch.TrainingResult;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.layer.LayerBackprop;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.util.Streams;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class DefaultNeuralNetworkTrainer<M extends Matrix<M>> implements NeuralNetworkTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private static final Logger LOGGER = LoggerFactory.getLogger(DefaultNeuralNetworkTrainer.class);

    private final List<LayerTrainer<M>> layerTrainers;
    private final NeuralNetworkTrainerConfig<M> config;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultNeuralNetworkTrainer(NeuralNetworkTrainerConfig<M> config, List<LayerTrainer<M>> layerTrainers) {
        this.layerTrainers = layerTrainers;
        this.config = config;
    }


// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetworkTrainer ---------------------

    @Override
    public NeuralNetwork<M> train(Dataset<M> trainingDataset) {
        if (layerTrainers.getFirst().inputSize() != trainingDataset.features().rowCount()) {
            throw new IllegalArgumentException(String.format("Dataset input must have input size %d.", layerTrainers.getFirst().inputSize()));
        }
        int epoch = 1;
        boolean continueTraining;
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestEpoch = -1;
        NeuralNetwork<M> bestNetwork = createNeuralNetwork();
        do {
            var result = config.trainingExecutor().execute(trainingDataset, this::performTrainingStep);
            var regularizationPenalty = layerTrainers.stream()
                    .mapToDouble(LayerTrainer::regularizationPenalty)
                    .sum();

            applyLayerUpdates(result.layerUpdates());

            var validationLoss = calculateValidationLoss();
            var outcome = new EpochOutcome(epoch, result.trainingLoss(), validationLoss, regularizationPenalty, result.trainingLoss() + regularizationPenalty);
            var score = config.scoringFunction().score(outcome);
            if (score > bestScore) {
                bestNetwork = createNeuralNetwork();
                bestScore = score;
                bestEpoch = epoch;
            }
            config.listener().onEpoch(outcome);
            continueTraining = !config.stoppingAdvisor().shouldStop(epoch, score);
            epoch++;
        } while (continueTraining);
        LOGGER.info("Training complete after {} epochs with an overall best score of {} at epoch {}.", epoch - 1, bestScore, bestEpoch);
        return bestNetwork;
    }

    private DefaultNeuralNetwork<M> createNeuralNetwork() {
        return new DefaultNeuralNetwork<>(layerTrainers.stream()
                .filter(LayerTrainer::isInference)
                .map(LayerTrainer::createLayer)
                .toList());
    }

    private void applyLayerUpdates(List<LayerUpdate<M>> layerUpdates) {
        Streams.zip(layerTrainers.stream(), layerUpdates.stream())
                .forEach(pair -> pair.left().applyUpdates(pair.right()));
    }

// -------------------------- OTHER METHODS --------------------------

    private double calculateValidationLoss() {
        if (config.validationDataset().features().isEmpty()) {
            return Double.NaN;
        }
        var inferred = layerTrainers.stream().reduce(config.validationDataset().features(), (M acc, LayerTrainer<M> layer) -> layer.forwardPass(acc).activations(), (a, _) -> a);
        return config.lossFunction().loss(inferred, config.validationDataset().labels());
    }

    private TrainingResult<M> performTrainingStep(Dataset<M> trainingDataset) {
        var forwardPassResult = performForwardPass(trainingDataset);
        var trainingLoss = config.lossFunction().loss(forwardPassResult.output(), trainingDataset.labels());
        var outputGradient = config.lossFunction().gradient(forwardPassResult.output(), trainingDataset.labels());
        var layerUpdates = new ArrayList<LayerUpdate<M>>();

        for (LayerBackprop<M> backProp : forwardPassResult.backProps()) {
            var result = backProp.apply(outputGradient);
            layerUpdates.addFirst(result.layerUpdate());
            outputGradient = result.outputGradient();
        }
        return new TrainingResult<>(trainingLoss, layerUpdates);
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

}
