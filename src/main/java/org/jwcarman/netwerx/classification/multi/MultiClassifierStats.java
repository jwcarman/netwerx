package org.jwcarman.netwerx.classification.multi;

public record MultiClassifierStats(
        double accuracy,
        double precision,
        double recall,
        double f1
) {

// -------------------------- STATIC METHODS --------------------------

    public static MultiClassifierStats of(int[] predicted, int[] actual, int numClasses) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual labels must have the same length.");
        }

        MultiConfusionMatrix matrix = new MultiConfusionMatrix(numClasses);
        for (int i = 0; i < predicted.length; i++) {
            matrix.increment(actual[i], predicted[i]);
        }

        double accuracy = matrix.accuracy();
        double precision = matrix.macroPrecision();
        double recall = matrix.macroRecall();
        double f1 = matrix.macroF1();

        return new MultiClassifierStats(accuracy, precision, recall, f1);
    }

}
