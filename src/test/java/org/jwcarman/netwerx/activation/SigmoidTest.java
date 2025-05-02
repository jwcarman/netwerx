package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class SigmoidTest {
    @Test
    void apply_shouldMapInputsToSigmoidValues() {
        var sigmoid = Activations.sigmoid();
        var input = Matrices.of(new double[][]{
                {-2.0, 0.0, 2.0}
        });
        var output = sigmoid.apply(input);

        assertThat(output.valueAt(0, 0)).isCloseTo(1.0 / (1 + Math.exp(2.0)), withinTolerance());
        assertThat(output.valueAt(0, 1)).isCloseTo(0.5, withinTolerance());
        assertThat(output.valueAt(0, 2)).isCloseTo(1.0 / (1 + Math.exp(-2.0)), withinTolerance());
    }

    @Test
    void derivative_shouldBeCorrectForEachElement() {
        var sigmoid = Activations.sigmoid();
        var input = Matrices.of(new double[][]{
                {-2.0, 0.0, 2.0}
        });
        var derived = sigmoid.derivative(input);

        // Compute expected values: sigmoid(x) * (1 - sigmoid(x))
        for (int i = 0; i < input.columnCount(); i++) {
            double x = input.valueAt(0, i);
            double s = 1.0 / (1 + Math.exp(-x));
            double expected = s * (1 - s);
            assertThat(derived.valueAt(0, i)).isCloseTo(expected, withinTolerance());
        }
    }
}