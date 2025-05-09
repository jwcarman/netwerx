package org.jwcarman.netwerx.loss;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class MeanAbsoluteErrorTest {

// ------------------------------ FIELDS ------------------------------

    private static final double EPSILON = 1e-6;

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testMeanAbsoluteErrorGradient() {
        var predictions = Matrices.of(new double[][]{
                {2.5, 0.0, 2.1}
        });
        var targets = Matrices.of(new double[][]{
                {3.0, -0.5, 2.0}
        });

        var gradient = LossFunctions.mae().gradient(predictions, targets);

        assertThat(gradient.rowCount()).isEqualTo(predictions.rowCount());
        assertThat(gradient.columnCount()).isEqualTo(predictions.columnCount());

        for (int row = 0; row < gradient.rowCount(); row++) {
            for (int col = 0; col < gradient.columnCount(); col++) {
                double diff = predictions.valueAt(row, col) - targets.valueAt(row, col);
                double expected = Math.signum(diff) / predictions.size();
                assertThat(gradient.valueAt(row, col)).isCloseTo(expected, within(EPSILON));
            }
        }
    }

    @Test
    void testMeanAbsoluteErrorLoss() {
        var predictions = Matrices.of(new double[][]{
                {2.5, 0.0, 2.1}
        });
        var targets = Matrices.of(new double[][]{
                {3.0, -0.5, 2.0}
        });

        var loss = LossFunctions.mae().loss(predictions, targets);

        // Expected mean absolute error
        double expectedLoss = (Math.abs(2.5 - 3.0) + Math.abs(0.0 - (-0.5)) + Math.abs(2.1 - 2.0)) / 3.0;

        assertThat(loss).isCloseTo(expectedLoss, within(EPSILON));
    }

    @Test
    void testPerfectPredictionsHaveZeroLoss() {
        var predictions = Matrices.of(new double[][]{
                {1.0, 2.0, 3.0}
        });
        var targets = Matrices.of(new double[][]{
                {1.0, 2.0, 3.0}
        });

        var loss = LossFunctions.mae().loss(predictions, targets);

        assertThat(loss).isCloseTo(0.0, within(EPSILON));
    }

}
