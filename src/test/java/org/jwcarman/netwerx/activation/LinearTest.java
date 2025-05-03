package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class LinearTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldReturnSameMatrix() {
        var linear = ActivationFunctions.linear();
        var input = Matrices.of(new double[][]{
                {1.5, -2.0, 0.0}
        });

        var result = linear.apply(input);

        assertThat(result.rowCount()).isEqualTo(input.rowCount());
        assertThat(result.columnCount()).isEqualTo(input.columnCount());
        for (int row = 0; row < input.rowCount(); row++) {
            for (int col = 0; col < input.columnCount(); col++) {
                assertThat(result.valueAt(row, col)).isCloseTo(input.valueAt(row, col), withinTolerance());
            }
        }
    }

    @Test
    void derivative_shouldReturnMatrixOfOnes() {
        var linear = ActivationFunctions.linear();
        var input = Matrices.of(new double[][]{
                {3.2, -1.1, 0.0}
        });

        var result = linear.derivative(input);

        assertThat(result.rowCount()).isEqualTo(input.rowCount());
        assertThat(result.columnCount()).isEqualTo(input.columnCount());
        for (int row = 0; row < result.rowCount(); row++) {
            for (int col = 0; col < result.columnCount(); col++) {
                assertThat(result.valueAt(row, col)).isCloseTo(1.0, withinTolerance());
            }
        }
    }

}