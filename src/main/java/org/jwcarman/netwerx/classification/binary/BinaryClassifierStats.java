package org.jwcarman.netwerx.classification.binary;

public record BinaryClassifierStats(
        double accuracy,
        double precision,
        double recall,
        double f1
) {

// -------------------------- STATIC METHODS --------------------------

    public static BinaryClassifierStats of(boolean[] predicted, boolean[] actual) {
        var matrix = ConfusionMatrix.of(actual, predicted);
        return new BinaryClassifierStats(matrix.accuracy(), matrix.precision(), matrix.recall(), matrix.f1());
    }

}
