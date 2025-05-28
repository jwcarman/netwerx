package org.jwcarman.netwerx.network;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.listener.TrainingListener;
import org.jwcarman.netwerx.loss.LossFunction;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.score.ScoringFunction;
import org.jwcarman.netwerx.stopping.StoppingAdvisor;

import java.util.Random;

public record NeuralNetworkTrainerConfig<M extends Matrix<M>>(
        LossFunction lossFunction,
        Dataset<M> validationDataset,
        int batchSize,
        int subBatchCount,
        Random random,
        ScoringFunction scoringFunction,
        StoppingAdvisor stoppingAdvisor,
        TrainingListener listener

) {

}
