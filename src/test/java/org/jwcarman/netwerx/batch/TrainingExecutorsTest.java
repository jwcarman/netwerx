package org.jwcarman.netwerx.batch;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;
import java.util.concurrent.Executors;

import static org.assertj.core.api.Assertions.assertThat;

class TrainingExecutorsTest {

    @Test
    void testMiniBatch() {
        var executor = TrainingExecutors.miniBatch();
        assertThat(executor).isInstanceOf(MiniBatchExecutor.class)
                .hasFieldOrPropertyWithValue("batchSize", MiniBatchExecutor.DEFAULT_BATCH_SIZE)
                .hasFieldOrPropertyWithValue("random", Randoms.defaultRandom())
                .hasFieldOrProperty("executor");
    }

    @Test
    void testMiniBatchWithBatchSize() {
        var executor = TrainingExecutors.miniBatch(12);
        assertThat(executor).isInstanceOf(MiniBatchExecutor.class)
                .hasFieldOrPropertyWithValue("batchSize", 12)
                .hasFieldOrPropertyWithValue("random", Randoms.defaultRandom())
                .hasFieldOrProperty("executor");
    }


    @Test
    void testMiniBatchWithCustomSetup() {
        var rand = new Random();
        var executorService = Executors.newCachedThreadPool();
        var executor = TrainingExecutors.miniBatch(12, rand, executorService);
        assertThat(executor).isInstanceOf(MiniBatchExecutor.class)
                .hasFieldOrPropertyWithValue("batchSize", 12)
                .hasFieldOrPropertyWithValue("random", rand)
                .hasFieldOrPropertyWithValue("executor", executorService);
    }


    @Test
    void testFullBatch() {
        var executor = TrainingExecutors.fullBatch();
        assertThat(executor).isInstanceOf(FullBatchExecutor.class);
    }

}