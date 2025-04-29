package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static java.lang.Math.exp;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class SwishTest {
    @Test
    void apply_shouldReturnExpectedSwishValues() {
        var swish = Activations.swish();  // default bias
        var input = new SimpleMatrix(1, 3, true, -2.0, 0.0, 2.0);
        var output = swish.apply(input);

        for (int i = 0; i < input.getNumCols(); i++) {
            double x = input.get(0, i);
            double sigmoid = 1.0 / (1.0 + exp(-x));
            double expected = x * sigmoid;
            assertThat(output.get(0, i)).isCloseTo(expected, within(1e-6));
        }
    }

    @Test
    void derivative_shouldReturnExpectedSwishGradient() {
        var swish = Activations.swish();
        var input = new SimpleMatrix(1, 3, true, -2.0, 0.0, 2.0);
        var derivative = swish.derivative(input);

        for (int i = 0; i < input.getNumCols(); i++) {
            double x = input.get(0, i);
            double sig = 1.0 / (1.0 + exp(-x));
            double sigPrime = sig * (1 - sig);
            double expected = sig + x * sigPrime;
            assertThat(derivative.get(0, i)).isCloseTo(expected, within(1e-6));
        }
    }

    @Test
    void customBiasConstructor_shouldNotAffectFunctionality() {
        var swish = Activations.swish(0.42);  // Custom initial bias
        var input = new SimpleMatrix(1, 1, true, 1.0);
        var output = swish.apply(input);

        double x = 1.0;
        double sigmoid = 1.0 / (1.0 + exp(-x));
        double expected = x * sigmoid;
        assertThat(output.get(0, 0)).isCloseTo(expected, within(1e-6));
    }
}