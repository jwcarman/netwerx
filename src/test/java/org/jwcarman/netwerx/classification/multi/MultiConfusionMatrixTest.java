package org.jwcarman.netwerx.classification.multi;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class MultiConfusionMatrixTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testMetricsForThreeClassConfusionMatrix() {
        MultiConfusionMatrix matrix = new MultiConfusionMatrix(3);

        // Simulate 9 total samples:
        matrix.increment(0, 0); // TP for class 0
        matrix.increment(0, 1); // FN for class 0
        matrix.increment(0, 2); // FN for class 0
        matrix.increment(1, 1); // TP for class 1
        matrix.increment(1, 2); // FN for class 1
        matrix.increment(2, 2); // TP for class 2
        matrix.increment(2, 0); // FN for class 2
        matrix.increment(2, 1); // FN for class 2
        matrix.increment(1, 0); // FN for class 1

        assertThat(matrix.totalSamples()).isEqualTo(9);
        assertThat(matrix.totalCorrect()).isEqualTo(3); // diagonal elements

        // Per-class metrics
        assertThat(matrix.tp(0)).isEqualTo(1);
        assertThat(matrix.fp(0)).isEqualTo(2);
        assertThat(matrix.fn(0)).isEqualTo(2);
        assertThat(matrix.precision(0)).isCloseTo(1.0 / 3.0, withinTolerance());
        assertThat(matrix.recall(0)).isCloseTo(1.0 / 3.0, withinTolerance());
        assertThat(matrix.f1(0)).isCloseTo(1.0 / 3.0, withinTolerance());

        assertThat(matrix.precision(1)).isCloseTo(1.0 / 3.0, withinTolerance());
        assertThat(matrix.recall(1)).isCloseTo(1.0 / 3.0, withinTolerance());
        assertThat(matrix.f1(1)).isCloseTo(1.0 / 3.0, withinTolerance());

        assertThat(matrix.precision(2)).isCloseTo(0.333333, withinTolerance());
        assertThat(matrix.recall(2)).isCloseTo(0.333333, withinTolerance());
        assertThat(matrix.f1(2)).isCloseTo(0.333333, withinTolerance());

        // Macro metrics
        assertThat(matrix.accuracy()).isCloseTo(3.0 / 9.0, withinTolerance());
        assertThat(matrix.macroPrecision()).isCloseTo(1.0 / 3.0, withinTolerance());
        assertThat(matrix.macroRecall()).isCloseTo(1.0 / 3.0, withinTolerance());
        assertThat(matrix.macroF1()).isCloseTo(1.0 / 3.0, withinTolerance());
    }

    @Test
    void testZeroParticipationClass() {
        MultiConfusionMatrix matrix = new MultiConfusionMatrix(3);

        matrix.increment(0, 0);
        matrix.increment(0, 0);
        matrix.increment(1, 1);
        matrix.increment(1, 1);

        assertThat(matrix.totalSamples()).isEqualTo(4);
        assertThat(matrix.totalCorrect()).isEqualTo(4);

        // Class 2 has no participation
        assertThat(matrix.tp(2)).isZero();
        assertThat(matrix.fp(2)).isZero();
        assertThat(matrix.fn(2)).isZero();
        assertThat(matrix.precision(2)).isEqualTo(0.0);
        assertThat(matrix.recall(2)).isEqualTo(0.0);
        assertThat(matrix.f1(2)).isEqualTo(0.0);
    }

    @Test
    void participatingClasses_shouldOnlyIncludeActiveClasses() {
        var matrix = new MultiConfusionMatrix(4); // 4 classes, class 3 will be unused

        // Class 0: TP, FN, FP
        matrix.increment(0, 0); // TP for class 0
        matrix.increment(1, 0); // FP for class 0
        matrix.increment(0, 1); // FN for class 0

        // Class 1: TP only
        matrix.increment(1, 1);

        // Class 2: no activity

        // Class 3: no activity — this should *not* be in participatingClasses()

        var participating = matrix.participatingClasses().boxed().toList();

        // Should only contain classes 0 and 1
        assertThat(participating).containsExactlyInAnyOrder(0, 1);
    }

    @Test
    void macroF1_shouldReturnZeroWhenPrecisionAndRecallAreZero() {
        var matrix = new MultiConfusionMatrix(3);
        // No calls to increment → precision and recall for all classes are 0

        assertThat(matrix.macroPrecision()).isEqualTo(0.0);
        assertThat(matrix.macroRecall()).isEqualTo(0.0);
        assertThat(matrix.macroF1()).isEqualTo(0.0);
    }
}