package org.jwcarman.netwerx.dataset;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.util.tuple.Pair;
import org.jwcarman.netwerx.util.tuple.Triple;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.jwcarman.netwerx.util.Lists.chunked;

/**
 * A column-oriented labeled dataset where each column represents a sample.
 * Features and labels must have the same number of columns.
 */
public record Dataset<M extends Matrix<M>>(M features, M labels) {

// --------------------------- CONSTRUCTORS ---------------------------

    public Dataset {
        if (features.columnCount() != labels.columnCount()) {
            throw new IllegalArgumentException("Feature and label column counts must match.");
        }

    }


    public static <M extends Matrix<M>> Dataset<M> forBinaryClassifier(M features, boolean[] labels) {
        return new Dataset<>(features, features.binaryClassifierLabels(labels));
    }

    public static <M extends Matrix<M>> Dataset<M> forMultiClassifier(M features, int classCount, int[] labels) {
        return new Dataset<>(features, features.multiClassifierClasses(classCount, labels));
    }

    public static <M extends Matrix<M>> Dataset<M> forRegressionModel(M features, double[] targets) {
        return new Dataset<>(features, features.regressionModelTargets(targets));
    }

    public static <M extends Matrix<M>> Dataset<M> forAutoencoder(M features) {
        return new Dataset<>(features, features);
    }

// -------------------------- OTHER METHODS --------------------------

    public Triple<Dataset<M>, Dataset<M>, Dataset<M>> split(Random random, double firstSplit, double secondSplit) {

        if (firstSplit <= 0.0 || firstSplit >= 1.0 || secondSplit <= 0.0 || secondSplit >= 1.0 || firstSplit + secondSplit >= 1.0) {
            throw new IllegalArgumentException("Split ratios must be between 0 and 1 and sum to less than 1.");
        }
        var indices = features.columnIndices();
        Collections.shuffle(indices, random);

        int firstSplitIndex = (int) (firstSplit * indices.size());
        int secondSplitIndex = (int) ((firstSplit + secondSplit) * indices.size());

        var firstIndices = indices.subList(0, firstSplitIndex);
        var secondIndices = indices.subList(firstSplitIndex, secondSplitIndex);
        var thirdIndices = indices.subList(secondSplitIndex, indices.size());

        var first = new Dataset<>(features.columnSelect(firstIndices), labels.columnSelect(firstIndices));
        var second = new Dataset<>(features.columnSelect(secondIndices), labels.columnSelect(secondIndices));
        var third = new Dataset<>(features.columnSelect(thirdIndices), labels.columnSelect(thirdIndices));

        return new Triple<>(first, second, third);
    }

    public Pair<Dataset<M>, Dataset<M>> split(Random random, double splitRatio) {
        if (splitRatio <= 0.0 || splitRatio >= 1.0) {
            throw new IllegalArgumentException("Split ratio must be between 0 and 1.");
        }

        var indices = features.columnIndices();
        Collections.shuffle(indices, random);

        int splitIndex = (int) (splitRatio * indices.size());
        var firstIndices = indices.subList(0, splitIndex);
        var secondIndices = indices.subList(splitIndex, indices.size());

        var first = new Dataset<>(features.columnSelect(firstIndices), labels.columnSelect(firstIndices));
        var second = new Dataset<>(features.columnSelect(secondIndices), labels.columnSelect(secondIndices));

        return new Pair<>(first, second);
    }

    /**
     * Returns a new dataset with the features and labels split into batches, in the order of the original dataset.
     *
     * @param batchSize the number of samples in each batch
     * @return a list of datasets, each containing a batch of samples
     */
    public List<Dataset<M>> batches(int batchSize) {
        return batches(features.columnIndices(), batchSize);
    }

    private List<Dataset<M>> batches(List<Integer> indices, int batchSize) {
        return chunked(indices, batchSize).stream()
                .map(chunk -> new Dataset<>(features.columnSelect(chunk), labels.columnSelect(chunk)))
                .toList();
    }

    /**
     * Returns a new dataset with the features and labels split into batches, in a random order.
     *
     * @param random    the random number generator to use for shuffling
     * @param batchSize the number of samples in each batch
     * @return a list of datasets, each containing a batch of samples
     */
    public List<Dataset<M>> batches(Random random, int batchSize) {
        var indices = features.columnIndices();
        Collections.shuffle(indices, random);
        return batches(indices, batchSize);
    }

    /**
     * Returns a new dataset with the features and labels shuffled in the same order.
     * This is useful for training models to ensure that samples are not in a specific order.
     *
     * @param random the random number generator to use for shuffling
     * @return a new shuffled dataset
     */
    public Dataset<M> shuffle(Random random) {
        var indices = features.columnIndices();
        Collections.shuffle(indices, random);
        return new Dataset<>(features.columnReorder(indices), labels.columnReorder(indices));
    }

}
