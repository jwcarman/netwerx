package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class LeakyReLUTest {

    @Test
    void apply_shouldMatchExpectedBehavior_withDefaults() {
        var activation = ActivationFunctions.leakyRelu(); // Uses default alpha = 0.01
        var input = Matrices.of(new double[][]{{-1.0, 2.0}});

        var output = activation.apply(input);

        assertThat(output.valueAt(0, 0)).isCloseTo(-0.01, withinTolerance()); // 0.01 * -1.0
        assertThat(output.valueAt(0, 1)).isCloseTo(2.0, withinTolerance());
    }

    @Test
    void derivative_shouldMatchExpectedBehavior_withDefaults() {
        var activation = ActivationFunctions.leakyRelu(); // Uses default alpha = 0.01
        var input = Matrices.of(new double[][]{{-1.0, 2.0}});

        var derivative = activation.derivative(input);

        assertThat(derivative.valueAt(0, 0)).isCloseTo(0.01, withinTolerance());  // Negative slope
        assertThat(derivative.valueAt(0, 1)).isCloseTo(1.0, withinTolerance());   // Positive slope
    }

    @Test
    void apply_shouldMatchExpectedBehavior_withCustomAlpha() {
        var activation = ActivationFunctions.leakyRelu(0.05);
        var input = Matrices.of(new double[][]{{-2.0, 3.0}});

        var output = activation.apply(input);

        assertThat(output.valueAt(0, 0)).isCloseTo(-0.1, withinTolerance()); // 0.05 * -2.0
        assertThat(output.valueAt(0, 1)).isCloseTo(3.0, withinTolerance());
    }

    @Test
    void derivative_shouldMatchExpectedBehavior_withCustomAlpha() {
        var activation = ActivationFunctions.leakyRelu(0.05);
        var input = Matrices.of(new double[][]{{-2.0, 3.0}});

        var derivative = activation.derivative(input);

        assertThat(derivative.valueAt(0, 0)).isCloseTo(0.05, withinTolerance());
        assertThat(derivative.valueAt(0, 1)).isCloseTo(1.0, withinTolerance());
    }
}