package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;

class SigmoidTest {
    @Test
    void apply_shouldMapInputsToSigmoidValues() {
        var sigmoid = Activations.sigmoid();
        var input = new SimpleMatrix(1, 3, true, -2.0, 0.0, 2.0);
        var output = sigmoid.apply(input);

        assertThat(output.get(0, 0)).isCloseTo(1.0 / (1 + Math.exp(2.0)), within(1e-6));
        assertThat(output.get(0, 1)).isCloseTo(0.5, within(1e-6));
        assertThat(output.get(0, 2)).isCloseTo(1.0 / (1 + Math.exp(-2.0)), within(1e-6));
    }

    @Test
    void derivative_shouldBeCorrectForEachElement() {
        var sigmoid = Activations.sigmoid();
        var input = new SimpleMatrix(1, 3, true, -2.0, 0.0, 2.0);
        var derived = sigmoid.derivative(input);

        // Compute expected values: sigmoid(x) * (1 - sigmoid(x))
        for (int i = 0; i < input.getNumCols(); i++) {
            double x = input.get(0, i);
            double s = 1.0 / (1 + Math.exp(-x));
            double expected = s * (1 - s);
            assertThat(derived.get(0, i)).isCloseTo(expected, within(1e-6));
        }
    }
}