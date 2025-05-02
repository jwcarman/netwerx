package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class ReLUTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldZeroOutNegatives() {
        var relu = Activations.relu();
        var input = Matrices.of(new double[][]{
                {-2.0, 0.0, 1.5}
        });

        var output = relu.apply(input);

        assertThat(output.rowCount()).isEqualTo(input.rowCount());
        assertThat(output.columnCount()).isEqualTo(input.columnCount());
        assertThat(output.valueAt(0, 0)).isCloseTo(0.0, withinTolerance());
        assertThat(output.valueAt(0, 1)).isCloseTo(0.0, withinTolerance());
        assertThat(output.valueAt(0, 2)).isCloseTo(1.5, withinTolerance());
    }

    @Test
    void derivative_shouldBeZeroForNegativesAndOneForPositives() {
        var relu = Activations.relu();
        var input = Matrices.of(new double[][]{
                {-1.0, 0.0, 2.5}
        });

        var derivative = relu.derivative(input);

        assertThat(derivative.rowCount()).isEqualTo(input.rowCount());
        assertThat(derivative.columnCount()).isEqualTo(input.columnCount());
        assertThat(derivative.valueAt(0, 0)).isCloseTo(0.0, withinTolerance()); // x < 0
        assertThat(derivative.valueAt(0, 1)).isCloseTo(0.0, withinTolerance()); // x == 0
        assertThat(derivative.valueAt(0, 2)).isCloseTo(1.0, withinTolerance()); // x > 0
    }

    @Test
    void constructor_withInitialBias_setsExpectedBias() {
        var relu = Activations.relu(0.01);
        var weight = relu.generateInitialBias();
        assertThat(weight).isCloseTo(0.01, withinTolerance());
    }

}
