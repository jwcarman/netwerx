package org.jwcarman.netwerx.loss;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.util.Streams;

import java.util.stream.Stream;

public class Losses {

// -------------------------- STATIC METHODS --------------------------

    public static Loss bce() {
        return new BinaryCrossEntropy();
    }

    public static Loss bce(double epsilon) {
        return new BinaryCrossEntropy(epsilon);
    }

    public static Loss mse() {
        return new MeanSquaredError();
    }

    public static Loss cce() {
        return new CategoricalCrossEntropy();
    }

    public static Loss cce(double epsilon) {
        return new CategoricalCrossEntropy(epsilon);
    }

    public static Loss mae() {
        return new MeanAbsoluteError();
    }

    public static Loss huber() {
        return new Huber();
    }

    public static Loss huber(double delta) {
        return new Huber(delta);
    }

    public static Loss hinge() {
        return new Hinge();
    }

    public static Loss logCosh() {
        return new LogCosh();
    }

    public static Loss weightedBce(double positiveWeight, double negativeWeight) {
        return new WeightedBinaryCrossEntropy(positiveWeight, negativeWeight);
    }

    public static Loss weightedBce(double positiveWeight, double negativeWeight, double epsilon) {
        return new WeightedBinaryCrossEntropy(positiveWeight, negativeWeight, epsilon);
    }

    public record PredictionTarget(int row, int col, double prediction, double target) {

    }

    public static <M extends Matrix<M>> Stream<PredictionTarget> predictionTargets(M predictions, M targets) {
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

// --------------------------- CONSTRUCTORS ---------------------------

    private Losses() {
        // Prevent instantiation
    }

}
