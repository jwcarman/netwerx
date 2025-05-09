package org.jwcarman.netwerx.loss;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class HingeTest {

    @Test
    void testHingeLossAndGradient() {
        var hinge = LossFunctions.hinge();

        var predictions = Matrices.of(new double[][] {
                {0.9, -0.8, 0.3}
        });

        var targets = Matrices.of(new double[][] {
                {1.0, -1.0, 1.0}
        });

        // Expected hinge loss manually:
        // sample 1: max(0, 1 - (1 * 0.9)) = 0.1
        // sample 2: max(0, 1 - (-1 * -0.8)) = 0.2
        // sample 3: max(0, 1 - (1 * 0.3)) = 0.7
        // Average: (0.1 + 0.2 + 0.7) / 3 = 0.333...

        double loss = hinge.loss(predictions, targets);
        assertThat(loss).isCloseTo(0.3333, within(0.0001));

        // Expected gradients:
        // sample 1: (1 * 0.9) < 1 ➔ grad = -1
        // sample 2: (-1 * -0.8) < 1 ➔ grad = 1
        // sample 3: (1 * 0.3) < 1 ➔ grad = -1
        var expectedGrad = Matrices.of(new double[][] {
                {-1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0}
        });

        var grad = hinge.gradient(predictions, targets);

        for (int col = 0; col < grad.columnCount(); col++) {
            assertThat(grad.valueAt(0, col)).isCloseTo(expectedGrad.valueAt(0, col), within(0.0001));
        }
    }

    @Test
    void gradient_shouldIncludeZeroCaseWhenMarginIsMet() {
        var hinge = new Hinge();
        var predictions = Matrices.of(new double[][]{
                {1.0, -1.0},
        });

        var targets = Matrices.of(new double[][]{
                {1.0, -1.0},
        });

        var gradient = hinge.gradient(predictions, targets);

        // First example: y * yHat = 1 * 1 = 1.0 → Not less than 1 → should be 0.0
        assertThat(gradient.valueAt(0, 0)).isEqualTo(0.0);
        // Second example: y * yHat = -1 * -1 = 1.0 → Not less than 1 → should be 0.0
        assertThat(gradient.valueAt(0, 1)).isEqualTo(0.0);
    }
}