package org.jwcarman.netwerx.loss;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static java.lang.Math.log;
import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class CategoricalCrossEntropyTest {
    @Test
    void testUnweightedCCE_Loss() {
        var loss = LossFunctions.cce();
        // 3 classes, 2 samples
        var predictions = Matrices.of(new double[][]{
                {0.7, 0.1},
                {0.2, 0.7},
                {0.1, 0.2}
        });
        var targets = Matrices.of(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0},
                {0.0, 0.0}
        });

        double result = loss.loss(predictions, targets);
        double expected = -((log(0.7) + log(0.7)) / 2.0);
        assertThat(result).isCloseTo(expected, withinTolerance());
    }

    @Test
    void testClampingBehaviorPreventsLog0() {
        var loss = LossFunctions.cce();
        var predictions = Matrices.of(new double[][]{
                {1e-20}, {0.0}, {1.0}
        });
        var targets = Matrices.of(new double[][]{
                {1.0}, {0.0}, {0.0}
        });

        double result = loss.loss(predictions, targets);

        double clamped = Math.max(CategoricalCrossEntropy.DEFAULT_EPSILON, 1e-20);
        double expected = -log(clamped);
        assertThat(result).isCloseTo(expected, withinTolerance());
    }

    @Test
    void testGradient() {
        var loss = LossFunctions.cce();
        var predictions = Matrices.of(new double[][]{
                {0.7}, {0.2}, {0.1}
        });
        var targets = Matrices.of(new double[][]{
                {1.0}, {0.0}, {0.0}
        });

        var grad = loss.gradient(predictions, targets);

        assertThat(grad.rowCount()).isEqualTo(3);
        assertThat(grad.columnCount()).isEqualTo(1);
        assertThat(grad.valueAt(0,0)).isCloseTo(-0.3, withinTolerance());
        assertThat(grad.valueAt(1,0)).isCloseTo(0.2, withinTolerance());
        assertThat(grad.valueAt(2,0)).isCloseTo(0.1, withinTolerance());
    }

    @Test
    void testCustomEpsilonIsUsed() {
        double customEpsilon = 1e-5;
        var loss = LossFunctions.cce(customEpsilon);

        var predictions = Matrices.of(new double[][]{
                {0.0}, {0.0}, {1.0}
        });
        var targets = Matrices.of(new double[][]{
                {1.0}, {0.0}, {0.0}
        });

        // Because predictions[0] = 0.0 and gets clamped to customEpsilon
        double expectedLoss = -log(customEpsilon);

        double result = loss.loss(predictions, targets);

        assertThat(result).isCloseTo(expectedLoss, withinTolerance());
    }
}