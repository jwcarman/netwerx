package org.jwcarman.netwerx.matrix;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.jwcarman.netwerx.util.Tolerances.DEFAULT_TOLERANCE;

public abstract class AbstractMatrixFactoryTestCase<M extends Matrix<M>> {

    protected abstract MatrixFactory<M> factory();


    private record Features(double feature1, double feature2, double feature3) {

    }

    @Test
    void testColumnOriented() {
        var features = IntStream.range(0, 10).mapToObj(i -> new Features(i, i * 2, i * 3)).toList();
        var matrix = factory().columnOriented(features, List.of(
                Features::feature1,
                Features::feature2,
                Features::feature3
        ));
        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(3);
        assertThat(matrix.columnCount()).isEqualTo(10);
        for (int i = 0; i < 10; i++) {
            assertThat(matrix.valueAt(0, i)).isEqualTo(i);
            assertThat(matrix.valueAt(1, i)).isEqualTo(i * 2);
            assertThat(matrix.valueAt(2, i)).isEqualTo(i * 3);
        }
    }

    @Test
    void testRowOriented() {
        var features = IntStream.range(0, 10).mapToObj(i -> new Features(i, i * 2, i * 3)).toList();
        var matrix = factory().rowOriented(features, List.of(
                Features::feature1,
                Features::feature2,
                Features::feature3
        ));
        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(10);
        assertThat(matrix.columnCount()).isEqualTo(3);
        for (int i = 0; i < 10; i++) {
            assertThat(matrix.valueAt(i, 0)).isEqualTo(i);
            assertThat(matrix.valueAt(i, 1)).isEqualTo(i * 2);
            assertThat(matrix.valueAt(i, 2)).isEqualTo(i * 3);
        }
    }

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
        M matrix1 = factory().from(2, 2, 1.0, 2.0, 3.0, 4.0);

        M matrix2 = factory().from(2, 2, 1.0, 2.0, 3.0, 4.0);

        assertThat(matrix1.isIdentical(matrix2, DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testIsNotIdentical() {
        M matrix1 = factory().from(2, 2, 1.0, 2.0, 3.0, 4.0);

        M matrix2 = factory().from(2, 2, 1.0, 2.0, 3.0, 5.0);

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
    public void testFromArray() {
        M matrix = factory().from(2, 2, 1.0, 2.0, 3.0, 4.0);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isEqualTo(1);
        assertThat(matrix.valueAt(0, 1)).isEqualTo(2);
        assertThat(matrix.valueAt(1, 0)).isEqualTo(3);
        assertThat(matrix.valueAt(1, 1)).isEqualTo(4);
    }

    @Test
    void testFromArrayWithInvalidDimensions() {
        MatrixFactory<M> factory = factory();

        assertThatThrownBy(() -> factory.from(3, 2, 1.0, 2.0, 3.0, 4.0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Invalid number of values (4) for the specified dimensions (3 x 2), expecting 6 values.");
    }

    @Test
    void testFrom2DArray() {
        double[][] data = {
                {1.0, 2.0},
                {3.0, 4.0}
        };

        M matrix = factory().from(data);

        assertThat(matrix).isNotNull();
        assertThat(matrix.rowCount()).isEqualTo(2);
        assertThat(matrix.columnCount()).isEqualTo(2);
        assertThat(matrix.valueAt(0, 0)).isEqualTo(1);
        assertThat(matrix.valueAt(0, 1)).isEqualTo(2);
        assertThat(matrix.valueAt(1, 0)).isEqualTo(3);
        assertThat(matrix.valueAt(1, 1)).isEqualTo(4);
    }

}
