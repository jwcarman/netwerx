package org.jwcarman.netwerx.classification.multi;

public record MultiClassifierStats(
        double accuracy,
        double precision,
        double recall,
        double f1
) {
    public static MultiClassifierStats of(int[] predicted, int[] actual, int numClasses) {
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException("Predicted and actual labels must have the same length.");
        }

        int[][] confusionMatrix = new int[numClasses][numClasses];

        int n = predicted.length;
        for (int i = 0; i < n; i++) {
            confusionMatrix[actual[i]][predicted[i]]++;
        }

        int correct = 0;
        for (int c = 0; c < numClasses; c++) {
            correct += confusionMatrix[c][c];
        }
        double accuracy = correct / (double) n;

        double totalPrecision = 0.0;
        double totalRecall = 0.0;
        int classesWithSamples = 0;

        for (int c = 0; c < numClasses; c++) {
            int tp = confusionMatrix[c][c];
            int fp = 0;
            int fn = 0;

            for (int i = 0; i < numClasses; i++) {
                if (i != c) {
                    fp += confusionMatrix[i][c];
                    fn += confusionMatrix[c][i];
                }
            }

            if (tp + fp > 0) {
                totalPrecision += tp / (double) (tp + fp);
            }

            if (tp + fn > 0) {
                totalRecall += tp / (double) (tp + fn);
            }

            if (tp + fp + fn > 0) {
                classesWithSamples++;
            }
        }

        // Avoid division by zero if there are no valid classes
        if (classesWithSamples == 0) {
            return new MultiClassifierStats(accuracy, 0.0, 0.0, 0.0);
        }

        double macroPrecision = totalPrecision / classesWithSamples;
        double macroRecall = totalRecall / classesWithSamples;
        double macroF1 = (macroPrecision + macroRecall == 0.0)
                ? 0.0
                : 2 * (macroPrecision * macroRecall) / (macroPrecision + macroRecall);

        return new MultiClassifierStats(accuracy, macroPrecision, macroRecall, macroF1);
    }
}