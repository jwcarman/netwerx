package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class ReLUTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldZeroOutNegatives() {
        var relu = Activations.relu();
        var input = new SimpleMatrix(new double[][]{
                {-2.0, 0.0, 1.5}
        });

        var output = relu.apply(input);

        assertThat(output.getNumRows()).isEqualTo(input.getNumRows());
        assertThat(output.getNumCols()).isEqualTo(input.getNumCols());
        assertThat(output.get(0, 0)).isCloseTo(0.0, withinTolerance());
        assertThat(output.get(0, 1)).isCloseTo(0.0, withinTolerance());
        assertThat(output.get(0, 2)).isCloseTo(1.5, withinTolerance());
    }

    @Test
    void derivative_shouldBeZeroForNegativesAndOneForPositives() {
        var relu = Activations.relu();
        var input = new SimpleMatrix(new double[][]{
                {-1.0, 0.0, 2.5}
        });

        var derivative = relu.derivative(input);

        assertThat(derivative.getNumRows()).isEqualTo(input.getNumRows());
        assertThat(derivative.getNumCols()).isEqualTo(input.getNumCols());
        assertThat(derivative.get(0, 0)).isCloseTo(0.0, withinTolerance()); // x < 0
        assertThat(derivative.get(0, 1)).isCloseTo(0.0, withinTolerance()); // x == 0
        assertThat(derivative.get(0, 2)).isCloseTo(1.0, withinTolerance()); // x > 0
    }

    @Test
    void constructor_withInitialBias_setsExpectedBias() {
        var relu = Activations.relu(0.01);
        var weight = relu.generateInitialBias();
        assertThat(weight).isCloseTo(0.01, withinTolerance());
    }

}
