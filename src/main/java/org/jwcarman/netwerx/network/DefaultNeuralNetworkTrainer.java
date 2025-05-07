package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.EpochOutcome;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.batch.TrainingExecutor;
import org.jwcarman.netwerx.batch.TrainingResult;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.layer.LayerBackprop;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;
import org.jwcarman.netwerx.util.Streams;

import java.util.ArrayList;
import java.util.List;

public class DefaultNeuralNetworkTrainer<M extends Matrix<M>> implements NeuralNetworkTrainer<M> {

// ------------------------------ FIELDS ------------------------------

    private final List<LayerTrainer<M>> layerTrainers;
    private final StoppingAdvisor stoppingAdvisor;
    private final LossFunction lossFunction;
    private final Dataset<M> validationDataset;
    private final TrainingExecutor<M> trainingExecutor;

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultNeuralNetworkTrainer(
            List<LayerTrainer<M>> layerTrainers,
            StoppingAdvisor stoppingAdvisor,
            LossFunction lossFunction,
            Dataset<M> validationDataset, TrainingExecutor<M> trainingExecutor) {
        this.layerTrainers = layerTrainers;
        this.stoppingAdvisor = stoppingAdvisor;
        this.lossFunction = lossFunction;
        this.validationDataset = validationDataset;
        this.trainingExecutor = trainingExecutor;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetworkTrainer ---------------------

    @Override
    public NeuralNetwork<M> train(Dataset<M> trainingDataset, TrainingObserver observer) {
        if(layerTrainers.getFirst().inputSize() != trainingDataset.features().rowCount()) {
            throw new IllegalArgumentException(String.format("Dataset input must have input size %d.", layerTrainers.getFirst().inputSize()));
        }
        int epoch = 1;
        boolean continueTraining;
        do {
            var result = trainingExecutor.execute(trainingDataset, this::performTrainingStep);
            var regularizationPenalty = layerTrainers.stream()
                    .mapToDouble(LayerTrainer::regularizationPenalty)
                    .sum();

            applyLayerUpdates(result.layerUpdates());

            var validationLoss = calculateValidationLoss();
            var outcome = new EpochOutcome(epoch, result.trainingLoss(), validationLoss, regularizationPenalty, result.trainingLoss() + regularizationPenalty);

            observer.onEpoch(outcome);
            continueTraining = !stoppingAdvisor.shouldStopAfter(outcome);
            epoch++;
        } while (continueTraining);
        final var layers = layerTrainers.stream()
                .filter(LayerTrainer::isInference)
                .map(LayerTrainer::createLayer)
                .toList();
        return new DefaultNeuralNetwork<>(layers);
    }

    private void applyLayerUpdates(List<LayerUpdate<M>> layerUpdates) {
        Streams.zip(layerTrainers.stream(), layerUpdates.stream())
                .forEach(pair -> pair.left().applyUpdates(pair.right()));
    }

// -------------------------- OTHER METHODS --------------------------

    private double calculateValidationLoss() {
        if (validationDataset.features().isEmpty()) {
            return Double.NaN;
        }
        var inferred = layerTrainers.stream().reduce(validationDataset.features(), (M acc, LayerTrainer<M> layer) -> layer.forwardPass(acc).activations(), (a, _) -> a);
        return lossFunction.loss(inferred, validationDataset.labels());
    }

    private TrainingResult<M> performTrainingStep(Dataset<M> trainingDataset) {
        var forwardPassResult = performForwardPass(trainingDataset);
        var trainingLoss = lossFunction.loss(forwardPassResult.output(), trainingDataset.labels());
        var outputGradient = lossFunction.gradient(forwardPassResult.output(), trainingDataset.labels());
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
