package org.jwcarman.netwerx.classification.binary;

public record ConfusionMatrix(int tp, int tn, int fp, int fn) {

// -------------------------- STATIC METHODS --------------------------

    public static ConfusionMatrix of(boolean[] predicted, boolean[] actual) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual labels must have the same length.");
        }

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        final int n = predicted.length;

        for (int i = 0; i < n; i++) {
            boolean p = predicted[i];
            boolean a = actual[i];

            if (p == a) {
                if (p) tp++;
                else tn++;
            } else {
                if (p) fp++;
                else fn++;
            }
        }

        return new ConfusionMatrix(tp, tn, fp, fn);
    }

// -------------------------- OTHER METHODS --------------------------

    public double accuracy() {
        return (tp + tn) / (double)(tp + tn + fp + fn);
    }

    public double f1() {
        double p = precision();
        double r = recall();
        return (p + r) == 0 ? 0.0 : 2 * (p * r) / (p + r);
    }

    public double precision() {
        return (tp + fp) == 0 ? 0.0 : tp / (double)(tp + fp);
    }

    public double recall() {
        return (tp + fn) == 0 ? 0.0 : tp / (double)(tp + fn);
    }

}
