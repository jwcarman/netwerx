package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.assertThat;

class BinaryCrossEntropyTest {

    @Test
    void computesLossAndGradientCorrectly() {
        var predictions = new SimpleMatrix(new double[][]{
                {0.9, 0.1, 0.8, 0.2}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 0.0, 1.0, 0.0}
        });

        var loss = Losses.bce();
        var result = loss.loss(predictions, targets);

        assertThat(result).isGreaterThan(0.0).isLessThan(1.0);

        var grad = loss.gradient(predictions, targets);
        assertThat(grad.getNumRows()).isEqualTo(1);
        assertThat(grad.getNumCols()).isEqualTo(4);
    }

    @Test
    void supportsCustomEpsilon() {
        var predictions = new SimpleMatrix(new double[][]{
                {1e-20, 1.0 - 1e-20}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 0.0}
        });

        var epsilon = 1e-10;
        var loss = Losses.bce(epsilon);

        var computedLoss = loss.loss(predictions, targets);

        assertThat(computedLoss)
                .isGreaterThan(0.0)
                .isLessThan(100.0);
    }

    @Test
    void gradientIsDifferenceBetweenPredictionsAndTargets() {
        var predictions = new SimpleMatrix(new double[][]{
                {0.7, 0.3}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 0.0}
        });

        var loss = Losses.bce();
        var grad = loss.gradient(predictions, targets);

        assertThat(grad.get(0, 0)).isCloseTo(-0.3, within(1e-6));
        assertThat(grad.get(0, 1)).isCloseTo(0.3, within(1e-6));
    }
}