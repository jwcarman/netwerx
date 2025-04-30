package org.jwcarman.netwerx.classification.multi;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Confusion matrix for multi-class classification.
 * Tracks predictions and calculates metrics like precision, recall, F1, and accuracy.
 */
public class MultiConfusionMatrix {

// ------------------------------ FIELDS ------------------------------

    private final int[][] matrix;

// --------------------------- CONSTRUCTORS ---------------------------

    public MultiConfusionMatrix(int numClasses) {
        this.matrix = new int[numClasses][numClasses];
    }

// -------------------------- OTHER METHODS --------------------------

    public double accuracy() {
        return totalCorrect() / (double) totalSamples();
    }

    public int totalCorrect() {
        return IntStream.range(0, size())
                .map(i -> matrix[i][i])
                .sum();
    }

    public int totalSamples() {
        return Arrays.stream(matrix)
                .flatMapToInt(Arrays::stream)
                .sum();
    }

    public double f1(int classIndex) {
        double p = precision(classIndex);
        double r = recall(classIndex);
        return (p + r) == 0.0 ? 0.0 : 2 * p * r / (p + r);
    }

    public double precision(int classIndex) {
        int tp = tp(classIndex);
        int fp = fp(classIndex);
        return (tp + fp) == 0 ? 0.0 : tp / (double) (tp + fp);
    }

    public int tp(int classIndex) {
        return matrix[classIndex][classIndex];
    }

    public int fp(int classIndex) {
        return IntStream.range(0, size())
                .filter(i -> i != classIndex)
                .map(i -> matrix[i][classIndex])
                .sum();
    }

    public double recall(int classIndex) {
        int tp = tp(classIndex);
        int fn = fn(classIndex);
        return (tp + fn) == 0 ? 0.0 : tp / (double) (tp + fn);
    }

    public int fn(int classIndex) {
        return IntStream.range(0, size())
                .filter(j -> j != classIndex)
                .map(j -> matrix[classIndex][j])
                .sum();
    }

// -------------------------- PUBLIC METHODS --------------------------

    public void increment(int actual, int predicted) {
        matrix[actual][predicted]++;
    }

    public double macroF1() {
        double p = macroPrecision();
        double r = macroRecall();
        return (p + r) == 0.0 ? 0.0 : 2 * p * r / (p + r);
    }

    public double macroPrecision() {
        return participatingClasses()
                .mapToDouble(this::precision)
                .average()
                .orElse(0.0);
    }

    public double macroRecall() {
        return participatingClasses()
                .mapToDouble(this::recall)
                .average()
                .orElse(0.0);
    }

    // ------------------------ PRIVATE HELPERS ------------------------

    private IntStream participatingClasses() {
        return IntStream.range(0, size())
                .filter(c -> tp(c) + fp(c) + fn(c) > 0);
    }

    public int size() {
        return matrix.length;
    }

}