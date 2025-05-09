package org.jwcarman.netwerx.loss;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class BinaryCrossEntropyTest {

    @Test
    void computesLossAndGradientCorrectly() {
        var predictions = Matrices.of(new double[][]{
                {0.9, 0.1, 0.8, 0.2}
        });
        var targets = Matrices.of(new double[][]{
                {1.0, 0.0, 1.0, 0.0}
        });

        var loss = LossFunctions.bce();
        var result = loss.loss(predictions, targets);

        assertThat(result).isGreaterThan(0.0).isLessThan(1.0);

        var grad = loss.gradient(predictions, targets);
        assertThat(grad.rowCount()).isEqualTo(1);
        assertThat(grad.columnCount()).isEqualTo(4);
    }

    @Test
    void supportsCustomEpsilon() {
        var predictions = Matrices.of(new double[][]{
                {1e-20, 1.0 - 1e-20}
        });
        var targets = Matrices.of(new double[][]{
                {1.0, 0.0}
        });

        var epsilon = 1e-10;
        var loss = LossFunctions.bce(epsilon);

        var computedLoss = loss.loss(predictions, targets);

        assertThat(computedLoss)
                .isGreaterThan(0.0)
                .isLessThan(100.0);
    }

    @Test
    void gradientIsDifferenceBetweenPredictionsAndTargets() {
        var predictions = Matrices.of(new double[][]{
                {0.7, 0.3}
        });
        var targets = Matrices.of(new double[][]{
                {1.0, 0.0}
        });

        var loss = LossFunctions.bce();
        var grad = loss.gradient(predictions, targets);

        assertThat(grad.valueAt(0, 0)).isCloseTo(-0.3, withinTolerance());
        assertThat(grad.valueAt(0, 1)).isCloseTo(0.3, withinTolerance());
    }
}