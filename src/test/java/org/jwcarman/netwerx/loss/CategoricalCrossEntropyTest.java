package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static java.lang.Math.log;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;

class CategoricalCrossEntropyTest {
    @Test
    void testUnweightedCCE_Loss() {
        var loss = Losses.cce();
        // 3 classes, 2 samples
        var predictions = new SimpleMatrix(new double[][]{
                {0.7, 0.1},
                {0.2, 0.7},
                {0.1, 0.2}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0},
                {0.0, 0.0}
        });

        double result = loss.loss(predictions, targets);
        double expected = -((log(0.7) + log(0.7)) / 2.0);
        assertThat(result).isCloseTo(expected, within(1e-6));
    }

    @Test
    void testClampingBehaviorPreventsLog0() {
        var loss = Losses.cce();
        var predictions = new SimpleMatrix(new double[][]{
                {1e-20}, {0.0}, {1.0}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0}, {0.0}, {0.0}
        });

        double result = loss.loss(predictions, targets);

        double clamped = Math.max(CategoricalCrossEntropy.DEFAULT_EPSILON, 1e-20);
        double expected = -log(clamped);
        assertThat(result).isCloseTo(expected, within(1e-6));
    }

    @Test
    void testGradient() {
        var loss = Losses.cce();
        var predictions = new SimpleMatrix(new double[][]{
                {0.7}, {0.2}, {0.1}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0}, {0.0}, {0.0}
        });

        var grad = loss.gradient(predictions, targets);

        assertThat(grad.getNumRows()).isEqualTo(3);
        assertThat(grad.getNumCols()).isEqualTo(1);
        assertThat(grad.get(0)).isCloseTo(-0.3, within(1e-6));
        assertThat(grad.get(1)).isCloseTo(0.2, within(1e-6));
        assertThat(grad.get(2)).isCloseTo(0.1, within(1e-6));
    }

    @Test
    void testCustomEpsilonIsUsed() {
        double customEpsilon = 1e-5;
        var loss = Losses.cce(customEpsilon);

        var predictions = new SimpleMatrix(new double[][]{
                {0.0}, {0.0}, {1.0}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0}, {0.0}, {0.0}
        });

        // Because predictions[0] = 0.0 and gets clamped to customEpsilon
        double expectedLoss = -log(customEpsilon);

        double result = loss.loss(predictions, targets);

        assertThat(result).isCloseTo(expectedLoss, within(1e-6));
    }
}