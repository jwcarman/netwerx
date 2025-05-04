package org.jwcarman.netwerx.data;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;

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
    public static <T,M extends Matrix<M>> M features(MatrixFactory<M> factory, List<T> data, Function<T, Double>... featureExtractors) {
        if (data.isEmpty()) {
            return factory.zeros(featureExtractors.length, 0);
        }
        final double[][] arr = new double[featureExtractors.length][data.size()];
        for (int i = 0; i < featureExtractors.length; i++) {
            final Function<T, Double> extractor = featureExtractors[i];
            for (int j = 0; j < data.size(); j++) {
                arr[i][j] = extractor.apply(data.get(j));
            }
        }

        return factory.from(arr);
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
     * Splits a list into training, validation, and test sets based on given ratios.
     *
     * @param data            the full dataset
     * @param trainRatio      the ratio of data to allocate to training (e.g. 0.7)
     * @param validationRatio the ratio of data for validation (e.g. 0.15)
     * @param testRatio       the ratio of data for testing (e.g. 0.15)
     * @param <T>             the type of data
     * @return a Triple of (train, validation, test) sets
     * @throws IllegalArgumentException if ratios do not sum to 1.0 ± ε
     */
    public static <T> Split<T> split(List<T> data, double trainRatio, double validationRatio, double testRatio, Random random) {
        final double total = trainRatio + validationRatio + testRatio;
        if (Math.abs(total - 1.0) > 1e-6) {
            throw new IllegalArgumentException("Ratios must sum to 1.0");
        }

        int totalSize = data.size();
        int trainSize = (int) Math.round(trainRatio * totalSize);
        int validationSize = (int) Math.round(validationRatio * totalSize);

        var shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, random);

        var train = shuffled.subList(0, trainSize);
        var validation = shuffled.subList(trainSize, trainSize + validationSize);
        var test = shuffled.subList(trainSize + validationSize, totalSize);

        return new Split<>(train, validation, test);
    }

    public record Split<T>(List<T> train, List<T> validation, List<T> test) {
    }

}
