package org.jwcarman.netwerx.matrix;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;
import static org.jwcarman.netwerx.util.Tolerances.DEFAULT_TOLERANCE;

public abstract class AbstractMatrixFactoryTestCase<M extends Matrix<M>> {

    protected abstract MatrixFactory<M> factory();


    @Test
    void testOnes() {
        M matrix = factory().ones(2, 2);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isEqualTo(1);
        assertThat(matrix.valueAt(0, 1)).isEqualTo(1);
        assertThat(matrix.valueAt(1, 0)).isEqualTo(1);
        assertThat(matrix.valueAt(1, 1)).isEqualTo(1);
    }

    @Test
    void testRandomWithRange() {
        Random random = new Random(42);
        double min = 0.0;
        double max = 1.0;

        M matrix = factory().random(2, 2, min, max, random);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isBetween(min, max);
        assertThat(matrix.valueAt(0, 1)).isBetween(min, max);
        assertThat(matrix.valueAt(1, 0)).isBetween(min, max);
        assertThat(matrix.valueAt(1, 1)).isBetween(min, max);
    }

    @Test
    void testGaussian() {
        Random random = new Random(42);
        double mean = 0.0;
        double stddev = 1.0;

        M matrix = factory().gaussian(2, 2, mean, stddev, random);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);

        assertThat(matrix.values()).allSatisfy(value -> {
            assertThat(value).isNotNaN();
            assertThat(value).isCloseTo(mean, within(2 * stddev));
        });
    }

    @Test
    void testIsIdentical() {
        M matrix1 = factory().from(new double[][]{
                {1, 2},
                {3, 4}
        });

        M matrix2 = factory().from(new double[][]{
                {1, 2},
                {3, 4}
        });

        assertThat(matrix1.isIdentical(matrix2, DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testIsNotIdentical() {
        M matrix1 = factory().from(new double[][]{
                {1, 2},
                {3, 4}
        });

        M matrix2 = factory().from(new double[][]{
                {1, 2},
                {3, 5}
        });

        assertThat(matrix1.isIdentical(matrix2, DEFAULT_TOLERANCE)).isFalse();
    }

    @Test
    void testRandom() {
        Random random = new Random(42);

        M matrix = factory().random(2, 2, random);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isNotNaN();
        assertThat(matrix.valueAt(0, 1)).isNotNaN();
        assertThat(matrix.valueAt(1, 0)).isNotNaN();
        assertThat(matrix.valueAt(1, 1)).isNotNaN();
    }
    @Test
    void testZeros() {
        M matrix = factory().zeros(2, 2);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isZero();
        assertThat(matrix.valueAt(0, 1)).isZero();
        assertThat(matrix.valueAt(1, 0)).isZero();
        assertThat(matrix.valueAt(1, 1)).isZero();
    }

    @Test
    public void testCreateWithData() {
        M matrix = factory().from(new double[][]{
                {1, 2},
                {3, 4}
        });

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isEqualTo(1);
        assertThat(matrix.valueAt(0, 1)).isEqualTo(2);
        assertThat(matrix.valueAt(1, 0)).isEqualTo(3);
        assertThat(matrix.valueAt(1, 1)).isEqualTo(4);
    }

}
