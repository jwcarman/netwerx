package org.jwcarman.netwerx.data;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Datasets {

// ------------------------------ FIELDS ------------------------------

    public static final boolean[] EMPTY_BINARY_LABELS = new boolean[0];
    public static final int[] EMPTY_MULTI_CLASS_LABELS = new int[0];

// -------------------------- STATIC METHODS --------------------------

    @SafeVarargs
    public static <T> SimpleMatrix features(List<T> data, Function<T, Double>... featureExtractors) {
        if (data.isEmpty()) {
            return new SimpleMatrix(featureExtractors.length, 0);
        }

        final var rows = featureExtractors.length;
        final var cols = data.size();
        final var features = new SimpleMatrix(rows, cols);

        for (int col = 0; col < cols; col++) {
            var item = data.get(col);
            for (int row = 0; row < rows; row++) {
                features.set(row, col, featureExtractors[row].apply(item));
            }
        }

        return features;
    }

    public static <T> double[] regressionLabels(List<T> data, Function<T, Double> labelExtractor) {
        if (data.isEmpty()) {
            return new double[0];
        }
        final double[] labels = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = labelExtractor.apply(data.get(i));
        }
        return labels;
    }

    public static <T> int[] multiClassLabels(List<T> data, Function<T, Integer> labelExtractor) {
        if (data.isEmpty()) {
            return EMPTY_MULTI_CLASS_LABELS;
        }
        final int[] labels = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = labelExtractor.apply(data.get(i));
        }
        return labels;
    }

    public static <T> boolean[] binaryLabels(List<T> data, Function<T, Boolean> labelExtractor) {
        if (data.isEmpty()) {
            return EMPTY_BINARY_LABELS;
        }
        final boolean[] labels = new boolean[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = labelExtractor.apply(data.get(i));
        }
        return labels;
    }

    /**
     * Normalizes a feature row in a SimpleMatrix to the range [0, 1].
     *
     * @param matrix The SimpleMatrix containing the features.
     * @param row    The row index of the feature to normalize
     */
    public static void normalizeFeature(SimpleMatrix matrix, int row) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        for (int col = 0; col < matrix.getNumCols(); col++) {
            double value = matrix.get(row, col);
            if (value < min) min = value;
            if (value > max) max = value;
        }

        double range = max - min;
        if (range == 0) {
            for (int col = 0; col < matrix.getNumCols(); col++) {
                matrix.set(row, col, 0.5);  // Uniform
            }
        } else {
            for (int col = 0; col < matrix.getNumCols(); col++) {
                double value = matrix.get(row, col);
                matrix.set(row, col, (value - min) / range);
            }
        }
    }

    /**
     * Splits a list of data points into a training set and a test set.
     *
     * @param data       The full dataset.
     * @param trainRatio The ratio of data to use for training (e.g., 0.8 for 80%).
     * @param random     The random number generator to use for shuffling.
     * @param <T>        The type of data points.
     * @return A Split containing the training and test sets.
     */
    public static <T> Split<T> split(List<T> data, float trainRatio, Random random) {
        if (trainRatio <= 0.0 || trainRatio >= 1.0) {
            throw new IllegalArgumentException("Training ratio must be between 0 and 1 (exclusive).");
        }

        var shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, random);

        var splitIndex = Math.round(shuffled.size() * trainRatio);
        var trainingSet = shuffled.subList(0, splitIndex);
        var testSet = shuffled.subList(splitIndex, shuffled.size());

        return new Split<>(trainingSet, testSet);
    }

// -------------------------- INNER CLASSES --------------------------

    public record Split<T>(List<T> trainingSet, List<T> testSet) {

    }

}
