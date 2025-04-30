package org.jwcarman.netwerx.regression;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class RegressionModelStatsTest {

    @Test
    void of_shouldComputeCorrectStats() {
        double[] predictions = {2.5, 0.0, 2.1, 7.8};
        double[] targets     = {3.0, -0.5, 2.0, 7.5};

        var stats = RegressionModelStats.of(predictions, targets);

        assertThat(stats.mse()).isCloseTo(0.15, withinTolerance());
        assertThat(stats.mae()).isCloseTo(0.35, withinTolerance());
        assertThat(stats.r2()).isCloseTo(0.9820895522, withinTolerance());
    }

    @Test
    void of_shouldReturnZeroR2WhenTotalVarianceIsZero() {
        double[] predictions = {1.0, 1.0, 1.0};
        double[] targets     = {1.0, 1.0, 1.0}; // No variance

        var stats = RegressionModelStats.of(predictions, targets);

        assertThat(stats.mse()).isZero();
        assertThat(stats.mae()).isZero();
        assertThat(stats.r2()).isZero(); // totalVariance == 0, fallback
    }
}