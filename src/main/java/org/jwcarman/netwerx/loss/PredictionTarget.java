package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.util.Streams;

import java.util.stream.Stream;

public record PredictionTarget(int row, int col, double prediction, double target) {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> Stream<PredictionTarget> of(M predictions, M targets) {
        return Streams.zip(
                predictions.elements(),
                targets.elements(),
                (pred, target) -> new PredictionTarget(
                        pred.row(),
                        pred.column(),
                        pred.value(),
                        target.value()
                )
        );
    }

}
