package org.jwcarman.netwerx.classification.binary;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class ConfusionMatrixTest {
    @Test
    void of_shouldComputeCorrectCounts() {
        boolean[] predicted = {true, false, true, false, true, false};
        boolean[] actual    = {true, false, false, true, false, false};

        ConfusionMatrix cm = ConfusionMatrix.of(predicted, actual);

        assertThat(cm.tp()).isEqualTo(1); // true positive
        assertThat(cm.tn()).isEqualTo(2); // true negative
        assertThat(cm.fp()).isEqualTo(2); // false positive
        assertThat(cm.fn()).isEqualTo(1); // false negative
    }

    @Test
    void metrics_shouldBeCorrectlyCalculated() {
        // tp = 30, tn = 50, fp = 10, fn = 10
        ConfusionMatrix cm = new ConfusionMatrix(30, 50, 10, 10);

        assertThat(cm.accuracy()).isCloseTo(0.8, withinTolerance());
        assertThat(cm.precision()).isCloseTo(0.75, withinTolerance());
        assertThat(cm.recall()).isCloseTo(0.75, withinTolerance());
        assertThat(cm.f1()).isCloseTo(0.75, withinTolerance());
    }

    @Test
    void metrics_shouldHandleZeroDivisionsGracefully() {
        ConfusionMatrix cm = new ConfusionMatrix(0, 0, 0, 0);

        assertThat(cm.accuracy()).isNaN(); // no samples
        assertThat(cm.precision()).isEqualTo(0.0);
        assertThat(cm.recall()).isEqualTo(0.0);
        assertThat(cm.f1()).isEqualTo(0.0);
    }

    @Test
    void metrics_shouldHandleAllPositivesCorrectly() {
        ConfusionMatrix cm = new ConfusionMatrix(10, 0, 0, 0);

        assertThat(cm.accuracy()).isCloseTo(1.0, withinTolerance());
        assertThat(cm.precision()).isCloseTo(1.0, withinTolerance());
        assertThat(cm.recall()).isCloseTo(1.0, withinTolerance());
        assertThat(cm.f1()).isCloseTo(1.0, withinTolerance());
    }
}