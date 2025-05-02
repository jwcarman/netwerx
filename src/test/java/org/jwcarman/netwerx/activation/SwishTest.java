package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static java.lang.Math.exp;
import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class SwishTest {
    @Test
    void apply_shouldReturnExpectedSwishValues() {
        var swish = Activations.swish();  // default bias
        var input = Matrices.of(new double[][]{
                {-2.0},
                {0.0},
                {2.0}
        });
        var output = swish.apply(input);

        for (int i = 0; i < input.columnCount(); i++) {
            double x = input.valueAt(0, i);
            double sigmoid = 1.0 / (1.0 + exp(-x));
            double expected = x * sigmoid;
            assertThat(output.valueAt(0, i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void derivative_shouldReturnExpectedSwishGradient() {
        var swish = Activations.swish();
        var input = Matrices.of(new double[][]{
                {-2.0},
                {0.0},
                {2.0}
        });
        var derivative = swish.derivative(input);

        for (int i = 0; i < input.columnCount(); i++) {
            double x = input.valueAt(0, i);
            double sig = 1.0 / (1.0 + exp(-x));
            double sigPrime = sig * (1 - sig);
            double expected = sig + x * sigPrime;
            assertThat(derivative.valueAt(0, i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void customBiasConstructor_shouldNotAffectFunctionality() {
        var swish = Activations.swish(0.42);  // Custom initial bias
        var input = Matrices.of(new double[][] {
                {1.0}
        });
        var output = swish.apply(input);

        double x = 1.0;
        double sigmoid = 1.0 / (1.0 + exp(-x));
        double expected = x * sigmoid;
        assertThat(output.valueAt(0, 0)).isCloseTo(expected, withinTolerance());
    }
}