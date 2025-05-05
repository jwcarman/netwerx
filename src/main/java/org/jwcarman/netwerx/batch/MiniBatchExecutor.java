package org.jwcarman.netwerx.batch;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.function.Function;

public class MiniBatchExecutor<M extends Matrix<M>> implements TrainingExecutor<M> {

// ------------------------------ FIELDS ------------------------------

    public static final int DEFAULT_BATCH_SIZE = 32;
    private final int batchSize;
    private final Random random;
    private final Executor executor;

// --------------------------- CONSTRUCTORS ---------------------------

    public MiniBatchExecutor() {
        this(DEFAULT_BATCH_SIZE, Randoms.defaultRandom(), Executors.newVirtualThreadPerTaskExecutor());
    }

    public MiniBatchExecutor(int batchSize) {
        this(batchSize, Randoms.defaultRandom(), Executors.newVirtualThreadPerTaskExecutor());
    }

    public MiniBatchExecutor(int batchSize, Random random, Executor executor) {
        this.batchSize = batchSize;
        this.random = random;
        this.executor = executor;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface TrainingExecutor ---------------------

    @Override
    public TrainingResult<M> execute(Dataset<M> dataset, Function<Dataset<M>, TrainingResult<M>> trainerFunction) {
        var futures = dataset.batches(random, batchSize).stream()
                .map(batch -> CompletableFuture.supplyAsync(() -> trainerFunction.apply(batch), executor))
                .toList();

        var results = futures.stream()
                .map(CompletableFuture::join)
                .toList();

        return TrainingResult.aggregate(results);
    }

}
