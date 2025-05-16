package org.jwcarman.netwerx.util.stats;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class StatsTest {

// ------------------------------ FIELDS ------------------------------

    public static final double[] EMPTY_ARRAY = new double[0];

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testCount() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.count()).isEqualTo(values.length);
    }

    @Test
    void testL2() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var sumOfSquares = Arrays.stream(values).map(v -> v*v).sum();
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.l2()).isCloseTo(Math.sqrt(sumOfSquares), withinTolerance());
    }

    @Test
    void testMax() {
        double[] values = {4.0, 1.0, 2.0, 3.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.max()).isEqualTo(5.0);
    }

    @Test
    void testMaxAbs() {
        double[] values = {-4.0, -1.0, 2.0, 3.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.maxAbs()).isEqualTo(5.0);
    }

    @Test
    void testMean() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.mean()).isEqualTo(3.0);
    }

    @Test
    void testMin() {
        double[] values = {4.0, 1.0, 2.0, 3.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.min()).isEqualTo(1.0);
    }

    @Test
    void testOfWithNoValues() {
        var values = Arrays.stream(EMPTY_ARRAY);
        assertThatThrownBy(() -> Stats.of(values))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void testWithParallelStream() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var stats = Stats.of(Arrays.stream(values).parallel());
        assertThat(stats.count()).isEqualTo(values.length);
        assertThat(stats.mean()).isEqualTo(3.0);
        assertThat(stats.min()).isEqualTo(1.0);
        assertThat(stats.max()).isEqualTo(5.0);
    }

    @Test
    void testStddev() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.stddev()).isCloseTo(1.4142135623730951, withinTolerance());
    }

    @Test
    void testVariance() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var stats = Stats.of(Arrays.stream(values));
        assertThat(stats.variance()).isCloseTo(2.0, withinTolerance());
    }

}