package org.jwcarman.netwerx.loss;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class HuberTest {

    @Test
    void testHuberLossAndGradientWithDefaultDelta() {
        var huber = LossFunctions.huber(); // Default delta = 1.0

        var predictions = Matrices.of(new double[][] {
                {2.5, 0.0, 1.0}
        });

        var targets = Matrices.of(new double[][] {
                {3.0, -1.0, 1.5}
        });

        double loss = huber.loss(predictions, targets);
        assertThat(loss).isCloseTo(0.25, within(0.0001));

        var expectedGrad = Matrices.of(new double[][] {
                {-0.5 / 3.0, 1.0 / 3.0, -0.5 / 3.0}
        });

        var grad = huber.gradient(predictions, targets);

        for (int col = 0; col < grad.columnCount(); col++) {
            assertThat(grad.valueAt(0, col)).isCloseTo(expectedGrad.valueAt(0, col), within(0.0001));
        }
    }

    @Test
    void testHuberLossAndGradientWithCustomDelta() {
        var huber = LossFunctions.huber(0.5); // Custom smaller delta

        var predictions = Matrices.of(new double[][] {
                {0.0, 2.0, 4.0}
        });

        var targets = Matrices.of(new double[][] {
                {0.2, 0.0, 6.0}
        });

        // Differences:
        // (0.0 - 0.2) = -0.2 ➔ inside delta (0.5)
        // (2.0 - 0.0) = 2.0 ➔ outside delta
        // (4.0 - 6.0) = -2.0 ➔ outside delta

        // Losses:
        // sample 1: 0.5 * (-0.2)^2 = 0.5 * 0.04 = 0.02
        // sample 2: 0.5 * (2.0 - 0.5) = 0.75
        // sample 3: 0.5 * (2.0 - 0.5) = 0.75
        // Total loss = (0.02 + 0.75 + 0.75) / 3 = 0.506666...

        double loss = huber.loss(predictions, targets);
        assertThat(loss).isCloseTo(0.59, within(0.0001));

        var grad = huber.gradient(predictions, targets);

        // Expected gradient:
        // sample 1: diff = -0.2 ➔ grad = -0.2
        // sample 2: diff = 2.0 ➔ grad = delta * sign(diff) = 0.5
        // sample 3: diff = -2.0 ➔ grad = -0.5

        double[] expectedGrads = {-0.2 / 3.0, 0.5 / 3.0, -0.5 / 3.0};

        for (int col = 0; col < grad.columnCount(); col++) {
            assertThat(grad.valueAt(0, col)).isCloseTo(expectedGrads[col], within(0.0001));
        }
    }
}