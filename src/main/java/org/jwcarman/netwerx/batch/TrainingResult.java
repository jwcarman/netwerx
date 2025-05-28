package org.jwcarman.netwerx.batch;

import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.List;
import java.util.stream.IntStream;

public record TrainingResult<M extends Matrix<M>>(int batchSize, double trainingLoss, List<LayerUpdate<M>> layerUpdates) {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> TrainingResult<M> aggregate(List<TrainingResult<M>> results) {
        if (results.isEmpty()) {
            throw new IllegalArgumentException("Cannot aggregate an empty list of training results.");
        }

        var totalBatchSize = results.stream()
                .mapToInt(TrainingResult::batchSize)
                .sum();

        var layerUpdates = IntStream.range(0, results.getFirst().layerUpdates().size())
                .boxed()
                .map(layer -> LayerUpdate.aggregate(results.stream()
                            .map(r -> r.layerUpdates().get(layer).scaled(r.batchSize()))
                            .toList()))
                .map(l -> l.scaled(1.0 / totalBatchSize))
                .toList();

        var trainingLoss = results.stream()
                .mapToDouble(r -> r.trainingLoss() * r.batchSize())
                .sum() / totalBatchSize;

        return new TrainingResult<>(totalBatchSize, trainingLoss, layerUpdates);
    }

}
