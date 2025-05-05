package org.jwcarman.netwerx.batch;

import org.jwcarman.netwerx.layer.LayerUpdate;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.List;
import java.util.stream.IntStream;

public record TrainingResult<M extends Matrix<M>>(double trainingLoss, List<LayerUpdate<M>> layerUpdates) {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> TrainingResult<M> aggregate(List<TrainingResult<M>> results) {
        if (results.isEmpty()) {
            throw new IllegalArgumentException("Cannot aggregate an empty list of training results.");
        }

        var layerUpdates = IntStream.range(0, results.getFirst().layerUpdates().size())
                .boxed()
                .map(layer -> LayerUpdate.aggregate(results.stream()
                            .map(r -> r.layerUpdates().get(layer))
                            .toList()))
                .toList();

        var trainingLoss = results.stream()
                .mapToDouble(TrainingResult::trainingLoss)
                .average()
                .orElse(Double.NaN);

        return new TrainingResult<>(trainingLoss, layerUpdates);
    }

}
