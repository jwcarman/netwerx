package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;

class LeakyReLUTest {

    @Test
    void apply_shouldMatchExpectedBehavior_withDefaults() {
        var activation = Activations.leakyRelu(); // Uses default alpha = 0.01
        var input = new SimpleMatrix(1, 2);
        input.set(0, 0, -1.0);
        input.set(0, 1, 2.0);

        var output = activation.apply(input);

        assertThat(output.get(0, 0)).isCloseTo(-0.01, within(1e-6)); // 0.01 * -1.0
        assertThat(output.get(0, 1)).isCloseTo(2.0, within(1e-6));
    }

    @Test
    void derivative_shouldMatchExpectedBehavior_withDefaults() {
        var activation = Activations.leakyRelu(); // Uses default alpha = 0.01
        var input = new SimpleMatrix(1, 2);
        input.set(0, 0, -1.0);  // pre-activation < 0
        input.set(0, 1, 2.0);   // pre-activation > 0

        var derivative = activation.derivative(input);

        assertThat(derivative.get(0, 0)).isCloseTo(0.01, within(1e-6));  // Negative slope
        assertThat(derivative.get(0, 1)).isCloseTo(1.0, within(1e-6));   // Positive slope
    }

    @Test
    void apply_shouldMatchExpectedBehavior_withCustomAlpha() {
        var activation = Activations.leakyRelu(0.01, 0.05);
        var input = new SimpleMatrix(1, 2);
        input.set(0, 0, -2.0);
        input.set(0, 1, 3.0);

        var output = activation.apply(input);

        assertThat(output.get(0, 0)).isCloseTo(-0.1, within(1e-6)); // 0.05 * -2.0
        assertThat(output.get(0, 1)).isCloseTo(3.0, within(1e-6));
    }

    @Test
    void derivative_shouldMatchExpectedBehavior_withCustomAlpha() {
        var activation = Activations.leakyRelu(0.01, 0.05);
        var input = new SimpleMatrix(1, 2);
        input.set(0, 0, -2.0);  // pre-activation < 0
        input.set(0, 1, 3.0);   // pre-activation > 0

        var derivative = activation.derivative(input);

        assertThat(derivative.get(0, 0)).isCloseTo(0.05, within(1e-6));
        assertThat(derivative.get(0, 1)).isCloseTo(1.0, within(1e-6));
    }
}