package org.jwcarman.netwerx.classification.binary;

public record BinaryClassifierStats(
        double accuracy,
        double precision,
        double recall,
        double f1
) {

// -------------------------- STATIC METHODS --------------------------

    public static BinaryClassifierStats of(boolean[] predicted, boolean[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual labels must have the same length.");
        }

        var tp = 0;
        var tn = 0;
        var fp = 0;
        var fn = 0;
        final var n = predicted.length;

        for (int i = 0; i < n; i++) {
            if (predicted[i]) {
                if (actual[i]) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (actual[i]) {
                    fn++;
                } else {
                    tn++;
                }
            }
        }

        double accuracy = (tp + tn) / (double) n;
        double precision = tp + fp == 0 ? 0.0 : tp / (double) (tp + fp);
        double recall = tp + fn == 0 ? 0.0 : tp / (double) (tp + fn);
        double f1 = precision + recall == 0 ? 0.0 : 2 * (precision * recall) / (precision + recall);

        return new BinaryClassifierStats(accuracy, precision, recall, f1);
    }

}
