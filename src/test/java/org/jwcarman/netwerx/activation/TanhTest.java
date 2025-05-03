package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class TanhTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldMapInputsToRangeNegativeOneToOne() {
        var tanh = ActivationFunctions.tanh();
        var input = Matrices.of(new double[][]{
                {-10.0},
                {0.0},
                {10.0}
        });
        var output = tanh.apply(input);

        assertThat(output.valueAt(0, 0)).isCloseTo(-1.0, withinTolerance());
        assertThat(output.valueAt(1, 0)).isCloseTo(0.0, withinTolerance());
        assertThat(output.valueAt(2, 0)).isCloseTo(1.0, withinTolerance());
    }

    @Test
    void derivative_shouldMatchExpectedBehavior() {
        var tanh = ActivationFunctions.tanh();
        var input = Matrices.of(new double[][]{
                {-10.0},
                {0.0},
                {10.0}
        });
        var output = tanh.apply(input);
        var derivative = tanh.derivative(input);

        // derivative of tanh(x) is 1 - tanh(x)^2
        for (int i = 0; i < input.rowCount(); i++) {
            double expected = 1.0 - Math.pow(output.valueAt(i, 0), 2);
            assertThat(derivative.valueAt(i, 0)).isCloseTo(expected, withinTolerance());
        }
    }

}
