package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class LogCoshTest {

// ------------------------------ FIELDS ------------------------------

    private static final double EPSILON = 1e-6;

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testLogCoshGradientSmallDifferences() {
        var predictions = new SimpleMatrix(new double[][]{
                {1.0, 2.0, 3.0}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.1, 1.9, 3.2}
        });

        var gradient = Losses.logCosh().gradient(predictions, targets);

        assertThat(gradient.getNumRows()).isEqualTo(predictions.getNumRows());
        assertThat(gradient.getNumCols()).isEqualTo(predictions.getNumCols());

        for (int row = 0; row < gradient.getNumRows(); row++) {
            for (int col = 0; col < gradient.getNumCols(); col++) {
                double diff = predictions.get(row, col) - targets.get(row, col);
                double expected = Math.tanh(diff);
                assertThat(gradient.get(row, col)).isCloseTo(expected, within(EPSILON));
            }
        }
    }

    @Test
    void testLogCoshLossSmallDifferences() {
        var predictions = new SimpleMatrix(new double[][]{
                {1.0, 2.0, 3.0}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.1, 1.9, 3.2}
        });

        var loss = Losses.logCosh().loss(predictions, targets);

        // Manually compute expected loss
        double expectedLoss = (
                Math.log(Math.cosh(1.0 - 1.1)) +
                        Math.log(Math.cosh(2.0 - 1.9)) +
                        Math.log(Math.cosh(3.0 - 3.2))
        ) / 3.0;

        assertThat(loss).isCloseTo(expectedLoss, within(EPSILON));
    }

}
