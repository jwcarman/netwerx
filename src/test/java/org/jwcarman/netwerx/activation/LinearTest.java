package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class LinearTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldReturnSameMatrix() {
        var linear = Activations.linear();
        var input = new SimpleMatrix(new double[][]{
                {1.5, -2.0, 0.0}
        });

        var result = linear.apply(input);

        assertThat(result.getNumRows()).isEqualTo(input.getNumRows());
        assertThat(result.getNumCols()).isEqualTo(input.getNumCols());
        for (int row = 0; row < input.getNumRows(); row++) {
            for (int col = 0; col < input.getNumCols(); col++) {
                assertThat(result.get(row, col)).isCloseTo(input.get(row, col), withinTolerance());
            }
        }
    }

    @Test
    void derivative_shouldReturnMatrixOfOnes() {
        var linear = Activations.linear();
        var input = new SimpleMatrix(new double[][]{
                {3.2, -1.1, 0.0}
        });

        var result = linear.derivative(input);

        assertThat(result.getNumRows()).isEqualTo(input.getNumRows());
        assertThat(result.getNumCols()).isEqualTo(input.getNumCols());
        for (int row = 0; row < result.getNumRows(); row++) {
            for (int col = 0; col < result.getNumCols(); col++) {
                assertThat(result.get(row, col)).isCloseTo(1.0, withinTolerance());
            }
        }
    }

}