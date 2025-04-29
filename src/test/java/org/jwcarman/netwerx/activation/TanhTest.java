package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class TanhTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldMapInputsToRangeNegativeOneToOne() {
        var tanh = Activations.tanh();
        var input = new SimpleMatrix(3, 1, true, -10.0, 0.0, 10.0);
        var output = tanh.apply(input);

        assertThat(output.get(0, 0)).isCloseTo(-1.0, within(1e-6));
        assertThat(output.get(1, 0)).isCloseTo(0.0, within(1e-6));
        assertThat(output.get(2, 0)).isCloseTo(1.0, within(1e-6));
    }

    @Test
    void derivative_shouldMatchExpectedBehavior() {
        var tanh = Activations.tanh();
        var input = new SimpleMatrix(3, 1, true, -10.0, 0.0, 10.0);
        var output = tanh.apply(input);
        var derivative = tanh.derivative(input);

        // derivative of tanh(x) is 1 - tanh(x)^2
        for (int i = 0; i < input.getNumRows(); i++) {
            double expected = 1.0 - Math.pow(output.get(i, 0), 2);
            assertThat(derivative.get(i, 0)).isCloseTo(expected, within(1e-6));
        }
    }

}
