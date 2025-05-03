package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.Dataset;
import org.jwcarman.netwerx.EpochOutcome;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.NeuralNetworkTrainer;
import org.jwcarman.netwerx.layer.LayerBackprop;
import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.layer.LayerTrainer;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;
import org.jwcarman.netwerx.util.Streams;

import java.util.ArrayList;
import java.util.List;

public class DefaultNeuralNetworkTrainer<M extends Matrix<M>> implements NeuralNetworkTrainer<M> {

    private final List<LayerTrainer<M>> layerTrainers;
    private final StoppingAdvisor stoppingAdvisor;
    private final LossFunction lossFunction;
    private final Dataset<M> validationDataset;

    public DefaultNeuralNetworkTrainer(List<LayerTrainer<M>> layerTrainers, StoppingAdvisor stoppingAdvisor, LossFunction lossFunction, Dataset<M> validationDataset) {
        this.layerTrainers = layerTrainers;
        this.stoppingAdvisor = stoppingAdvisor;
        this.lossFunction = lossFunction;
        this.validationDataset = validationDataset;
    }

    @Override
    public NeuralNetwork<M> train(Dataset<M> trainingDataset, TrainingObserver observer) {
        int epoch = 1;
        boolean continueTraining;
        do {
            var result = performTrainingStep(trainingDataset);
            Streams.zip(layerTrainers.stream(), result.layerUpdates().stream())
                    .forEach(pair -> pair.left().applyUpdates(pair.right()));
            var outcome = new EpochOutcome(epoch, result.trainingLoss(), result.validationLoss());
            observer.onEpoch(outcome);
            continueTraining = !stoppingAdvisor.shouldStopAfter(outcome);
            epoch++;
        } while (continueTraining);
        final var layers = layerTrainers.stream()
                .map(LayerTrainer::createLayer)
                .toList();
        return new DefaultNeuralNetwork<>(layers);
    }

    record TrainingStepResult<M extends Matrix<M>>(double trainingLoss, double validationLoss, List<LayerUpdate<M>> layerUpdates) {
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
        return new TrainingStepResult<>(trainingLoss, Double.NaN, layerUpdates);
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

    private record ForwardPassResult<M extends Matrix<M>>(M output, ArrayList<LayerBackprop<M>> backProps) {
    }
}
