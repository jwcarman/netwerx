package org.jwcarman.netwerx.batch;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.Random;
import java.util.concurrent.Executor;

public class TrainingExecutors {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> TrainingExecutor<M> fullBatch() {
        return new FullBatchExecutor<>();
    }

    public static <M extends Matrix<M>> TrainingExecutor<M> miniBatch() {
        return new MiniBatchExecutor<>();
    }

    public static <M extends Matrix<M>> TrainingExecutor<M> miniBatch(int batchSize) {
        return new MiniBatchExecutor<>(batchSize);
    }

    public static <M extends Matrix<M>> TrainingExecutor<M> miniBatch(int batchSize, Random random, Executor executor) {
        return new MiniBatchExecutor<>(batchSize, random, executor);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private TrainingExecutors() {
        // Prevent instantiation
    }

}
