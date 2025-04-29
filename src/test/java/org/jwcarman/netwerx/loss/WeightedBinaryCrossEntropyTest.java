package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class WeightedBinaryCrossEntropyTest {
    @Test
    void computesLossAndGradientCorrectly() {
        // Binary labels (0 and 1)
        var predictions = new SimpleMatrix(new double[][]{
                {0.9, 0.2, 0.8, 0.1}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 0.0, 1.0, 0.0}
        });

        var loss = Losses.weightedBce(2.0, 1.0);  // Heavier weight for positive labels
        var result = loss.loss(predictions, targets);

        // This expected value is just a ballpark for this example
        assertThat(result).isBetween(0.1, 1.0);

        var grad = loss.gradient(predictions, targets);
        assertThat(grad.getNumRows()).isEqualTo(1);
        assertThat(grad.getNumCols()).isEqualTo(4);
    }

    @Test
    void supportsCustomEpsilon() {
        // Force very small predictions to test epsilon clamping
        var predictions = new SimpleMatrix(new double[][]{
                {1e-20, 1.0 - 1e-20}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 0.0}
        });

        double epsilon = 1e-10;
        var loss = Losses.weightedBce(1.0, 1.0, epsilon);

        double computedLoss = loss.loss(predictions, targets);

        // Clamping should prevent log(0) and keep loss finite
        assertThat(computedLoss)
                .isGreaterThan(0.0)
                .isLessThan(100.0);
    }
}