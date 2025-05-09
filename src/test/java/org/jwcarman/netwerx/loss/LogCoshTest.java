package org.jwcarman.netwerx.loss;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class LogCoshTest {

// ------------------------------ FIELDS ------------------------------

    private static final double EPSILON = 1e-6;

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testLogCoshGradientSmallDifferences() {
        var predictions = Matrices.of(new double[][]{
                {1.0, 2.0, 3.0}
        });
        var targets = Matrices.of(new double[][]{
                {1.1, 1.9, 3.2}
        });

        var gradient = LossFunctions.logCosh().gradient(predictions, targets);

        assertThat(gradient.rowCount()).isEqualTo(predictions.rowCount());
        assertThat(gradient.columnCount()).isEqualTo(predictions.columnCount());

        for (int row = 0; row < gradient.rowCount(); row++) {
            for (int col = 0; col < gradient.columnCount(); col++) {
                double diff = predictions.valueAt(row, col) - targets.valueAt(row, col);
                double expected = Math.tanh(diff);
                assertThat(gradient.valueAt(row, col)).isCloseTo(expected, within(EPSILON));
            }
        }
    }

    @Test
    void testLogCoshLossSmallDifferences() {
        var predictions = Matrices.of(new double[][]{
                {1.0, 2.0, 3.0}
        });
        var targets = Matrices.of(new double[][]{
                {1.1, 1.9, 3.2}
        });

        var loss = LossFunctions.logCosh().loss(predictions, targets);

        // Manually compute expected loss
        double expectedLoss = (
                Math.log(Math.cosh(1.0 - 1.1)) +
                        Math.log(Math.cosh(2.0 - 1.9)) +
                        Math.log(Math.cosh(3.0 - 3.2))
        ) / 3.0;

        assertThat(loss).isCloseTo(expectedLoss, within(EPSILON));
    }

}
