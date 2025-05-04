package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.EpochOutcome;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
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

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultNeuralNetworkTrainer(
            List<LayerTrainer<M>> layerTrainers,
            StoppingAdvisor stoppingAdvisor,
            LossFunction lossFunction,
            Dataset<M> validationDataset) {
        this.layerTrainers = layerTrainers;
        this.stoppingAdvisor = stoppingAdvisor;
        this.lossFunction = lossFunction;
        this.validationDataset = validationDataset;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetworkTrainer ---------------------

    @Override
    public NeuralNetwork<M> train(Dataset<M> trainingDataset, TrainingObserver observer) {
        int epoch = 1;
        boolean continueTraining;
        do {
            var result = performTrainingStep(trainingDataset);
            var regularizationPenalty = layerTrainers.stream()
                    .mapToDouble(LayerTrainer::regularizationPenalty)
                    .sum();

            Streams.zip(layerTrainers.stream(), result.layerUpdates().stream())
                    .forEach(pair -> pair.left().applyUpdates(pair.right()));

            var validationLoss = calculateValidationLoss();
            var outcome = new EpochOutcome(epoch, result.trainingLoss(), validationLoss, regularizationPenalty, result.trainingLoss() + regularizationPenalty);


            observer.onEpoch(outcome);
            continueTraining = !stoppingAdvisor.shouldStopAfter(outcome);
            epoch++;
        } while (continueTraining);
        final var layers = layerTrainers.stream()
                .map(LayerTrainer::createLayer)
                .toList();
        return new DefaultNeuralNetwork<>(layers);
    }

// -------------------------- OTHER METHODS --------------------------

    private double calculateValidationLoss() {
        if (validationDataset.inputs().isEmpty()) {
            return Double.NaN;
        }
        var inferred = layerTrainers.stream().reduce(validationDataset.inputs(), (M acc, LayerTrainer<M> layer) -> layer.forwardPass(acc).activations(), (a, _) -> a);
        return lossFunction.loss(inferred, validationDataset.outputs());
    }

    private TrainingStepResult<M> performTrainingStep(Dataset<M> trainingDataset) {
        var forwardPassResult = performForwardPass(trainingDataset);
        var trainingLoss = lossFunction.loss(forwardPassResult.output(), trainingDataset.outputs());
        var outputGradient = lossFunction.gradient(forwardPassResult.output(), trainingDataset.outputs());
        var layerUpdates = new ArrayList<LayerUpdate<M>>();

        for (LayerBackprop<M> backProp : forwardPassResult.backProps()) {
            var result = backProp.apply(outputGradient);
            layerUpdates.addFirst(result.layerUpdate());
            outputGradient = result.outputGradient();
        }
        return new TrainingStepResult<>(trainingLoss, layerUpdates);
    }

    private ForwardPassResult<M> performForwardPass(Dataset<M> trainingDataset) {
        M activations = trainingDataset.inputs();
        var backProps = new ArrayList<LayerBackprop<M>>();
        for (LayerTrainer<M> trainer : layerTrainers) {
            var bp = trainer.forwardPass(activations);
            backProps.addFirst(bp);
            activations = bp.activations();
        }
        return new ForwardPassResult<>(activations, backProps);
    }

// -------------------------- INNER CLASSES --------------------------

    record TrainingStepResult<M extends Matrix<M>>(double trainingLoss, List<LayerUpdate<M>> layerUpdates) {

    }

    private record ForwardPassResult<M extends Matrix<M>>(M output, ArrayList<LayerBackprop<M>> backProps) {

    }

}
