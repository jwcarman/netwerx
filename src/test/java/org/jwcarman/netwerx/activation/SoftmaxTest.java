package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class SoftmaxTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldBeNumericallyStableForLargeValues() {
        var softmax = Activations.softmax();
        var input = Matrices.of(new double[][]{
                {1.0},
                {2.0},
                {100.0}
        });
        var output = softmax.apply(input);

        // Find index of max value
        int maxIdx = 0;
        double maxVal = output.valueAt(0, 0);
        for (int i = 1; i < output.rowCount(); i++) {
            if (output.valueAt(i, 0) > maxVal) {
                maxVal = output.valueAt(i, 0);
                maxIdx = i;
            }
        }

        // Confirm maxIdx corresponds to input value 100.0 (row 2)
        assertThat(maxIdx).isEqualTo(2);

        // Confirm softmax probabilities still sum to 1
        double sum = output.sum();
        assertThat(sum).isCloseTo(1.0, withinTolerance());
    }

    @Test
    void apply_shouldOutputProbabilitiesSummingToOnePerColumn() {
        var softmax = Activations.softmax();
        var input = Matrices.of(new double[][]{
                {1.0, 5.0},
                {2.0, 4.0},
                {3.0, 6.0}
        });
        var output = softmax.apply(input);

        for (int col = 0; col < output.columnCount(); col++) {
            assertThat(output.columnSum(col)).isCloseTo(1.0, withinTolerance());
        }
    }

    @Test
    void apply_shouldOutputSameShape() {
        var softmax = Activations.softmax();
        var input = Matrices.of(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
        });
        var output = softmax.apply(input);

        assertThat(output.rowCount()).isEqualTo(3);
        assertThat(output.columnCount()).isEqualTo(2);
    }

}
