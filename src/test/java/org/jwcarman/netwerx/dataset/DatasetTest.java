package org.jwcarman.netwerx.dataset;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.util.Randoms;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class DatasetTest {

    @Test
    void testSplitIntoPair() {
        var factory = new EjmlMatrixFactory();
        var features = factory.filled(4, 4, 1.0);
        var labels = factory.filled(1, 4, 2.0);
        var dataset = new Dataset<>(features, labels);

        var rng = Randoms.defaultRandom();
        assertThatThrownBy(() -> dataset.split(rng,-1.0))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> dataset.split(rng,2.0))
                .isInstanceOf(IllegalArgumentException.class);

        // Split the dataset
        var split = dataset.split(rng, 0.5);
        assertNotNull(split);

        // Check left side
        assertThat(split.left().features().columnCount()).isEqualTo(2);
        assertThat(split.left().features().rowCount()).isEqualTo(4);
        assertThat(split.left().labels().columnCount()).isEqualTo(2);
        assertThat(split.left().labels().rowCount()).isEqualTo(1);

        // Check right side
        assertThat(split.right().features().columnCount()).isEqualTo(2);
        assertThat(split.right().features().rowCount()).isEqualTo(4);
        assertThat(split.right().labels().columnCount()).isEqualTo(2);
        assertThat(split.right().labels().rowCount()).isEqualTo(1);
    }
    
    @Test
    void testSplitIntoTriple() {
        var factory = new EjmlMatrixFactory();
        var features = factory.filled(4, 4, 1.0);
        var labels = factory.filled(1, 4, 2.0);
        var dataset = new Dataset<>(features, labels);

        var rng = Randoms.defaultRandom();
        assertThatThrownBy(() -> dataset.split(rng,-1.0, 0.5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> dataset.split(rng,1.5, 0.5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> dataset.split(rng,0.5,-0.5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> dataset.split(rng,0.5,1.5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> dataset.split(rng,0.5,0.7))
                .isInstanceOf(IllegalArgumentException.class);
        // Split the dataset
        var split = dataset.split(rng, 0.5, 0.25);
        assertNotNull(split);

        // Check first part
        assertThat(split.first().features().columnCount()).isEqualTo(2);
        assertThat(split.first().features().rowCount()).isEqualTo(4);
        assertThat(split.first().labels().columnCount()).isEqualTo(2);
        assertThat(split.first().labels().rowCount()).isEqualTo(1);

        // Check second part
        assertThat(split.second().features().columnCount()).isEqualTo(1);
        assertThat(split.second().features().rowCount()).isEqualTo(4);
        assertThat(split.second().labels().columnCount()).isEqualTo(1);
        assertThat(split.second().labels().rowCount()).isEqualTo(1);

        // Check third part
        assertThat(split.third().features().columnCount()).isEqualTo(1);
        assertThat(split.third().features().rowCount()).isEqualTo(4);
        assertThat(split.third().labels().columnCount()).isEqualTo(1);
    }

    @Test
    void testConstructorWithMismatchedColumns() {
        var factory = new EjmlMatrixFactory();
        var features = factory.zeros(2, 2);
        var labels = factory.zeros(1, 3);


        assertThatThrownBy(() -> new Dataset<>(features, labels))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void testBatchesInOrder() {
        var factory = new EjmlMatrixFactory();
        var features = factory.filled(4, 4, 1.0);
        var labels = factory.filled(1, 4, 2.0);
        var dataset = new Dataset<>(features, labels);

        var batches = dataset.batches(2);
        assertThat(batches).hasSize(2);

        // Check first batch
        var firstBatch = batches.getFirst();
        assertThat(firstBatch.features().rowCount()).isEqualTo(4);
        assertThat(firstBatch.features().columnCount()).isEqualTo(2);
        assertThat(firstBatch.labels().rowCount()).isEqualTo(1);
        assertThat(firstBatch.labels().columnCount()).isEqualTo(2);

        // Check second batch
        var secondBatch = batches.get(1);
        assertThat(secondBatch.features().rowCount()).isEqualTo(4);
        assertThat(secondBatch.features().columnCount()).isEqualTo(2);
        assertThat(secondBatch.labels().rowCount()).isEqualTo(1);
        assertThat(secondBatch.labels().columnCount()).isEqualTo(2);
    }

    @Test
    void testBatchesRandom() {
        var factory = new EjmlMatrixFactory();
        var features = factory.filled(4, 4, 1.0);
        var labels = factory.filled(1, 4, 2.0);
        var dataset = new Dataset<>(features, labels);

        // Create batches with a different batch size
        var batches = dataset.batches(Randoms.defaultRandom(), 2);
        assertThat(batches).hasSize(2);

        // Check first batch
        var firstBatch = batches.getFirst();
        assertThat(firstBatch.features().rowCount()).isEqualTo(4);
        assertThat(firstBatch.features().columnCount()).isEqualTo(2);
        assertThat(firstBatch.labels().rowCount()).isEqualTo(1);
        assertThat(firstBatch.labels().columnCount()).isEqualTo(2);

        // Check second batch
        var secondBatch = batches.get(1);
        assertThat(secondBatch.features().rowCount()).isEqualTo(4);
        assertThat(secondBatch.features().columnCount()).isEqualTo(2);
        assertThat(secondBatch.labels().rowCount()).isEqualTo(1);
        assertThat(secondBatch.labels().columnCount()).isEqualTo(2);
    }

    @Test
    void testShuffle() {
        var factory = new EjmlMatrixFactory();
        var features = factory.filled(4, 4, 1.0);
        var labels = factory.filled(1, 4, 2.0);
        var dataset = new Dataset<>(features, labels);

        // Shuffle the dataset
        var shuffledDataset = dataset.shuffle(Randoms.defaultRandom());

        // Check that the features and labels are still the same size
        assertThat(shuffledDataset.features().rowCount()).isEqualTo(features.rowCount());
        assertThat(shuffledDataset.labels().rowCount()).isEqualTo(labels.rowCount());

        // Check that the data is shuffled
        assertNotEquals(dataset.features(), shuffledDataset.features());
        assertNotEquals(dataset.labels(), shuffledDataset.labels());
    }
}