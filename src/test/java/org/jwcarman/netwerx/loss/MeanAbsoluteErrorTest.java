package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class MeanAbsoluteErrorTest {

// ------------------------------ FIELDS ------------------------------

    private static final double EPSILON = 1e-6;

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testMeanAbsoluteErrorGradient() {
        var predictions = new SimpleMatrix(new double[][]{
                {2.5, 0.0, 2.1}
        });
        var targets = new SimpleMatrix(new double[][]{
                {3.0, -0.5, 2.0}
        });

        var gradient = Losses.mae().gradient(predictions, targets);

        assertThat(gradient.getNumRows()).isEqualTo(predictions.getNumRows());
        assertThat(gradient.getNumCols()).isEqualTo(predictions.getNumCols());

        for (int row = 0; row < gradient.getNumRows(); row++) {
            for (int col = 0; col < gradient.getNumCols(); col++) {
                double diff = predictions.get(row, col) - targets.get(row, col);
                double expected = Math.signum(diff) / (predictions.getNumRows() * predictions.getNumCols());
                assertThat(gradient.get(row, col)).isCloseTo(expected, within(EPSILON));
            }
        }
    }

    @Test
    void testMeanAbsoluteErrorLoss() {
        var predictions = new SimpleMatrix(new double[][]{
                {2.5, 0.0, 2.1}
        });
        var targets = new SimpleMatrix(new double[][]{
                {3.0, -0.5, 2.0}
        });

        var loss = Losses.mae().loss(predictions, targets);

        // Expected mean absolute error
        double expectedLoss = (Math.abs(2.5 - 3.0) + Math.abs(0.0 - (-0.5)) + Math.abs(2.1 - 2.0)) / 3.0;

        assertThat(loss).isCloseTo(expectedLoss, within(EPSILON));
    }

    @Test
    void testPerfectPredictionsHaveZeroLoss() {
        var predictions = new SimpleMatrix(new double[][]{
                {1.0, 2.0, 3.0}
        });
        var targets = new SimpleMatrix(new double[][]{
                {1.0, 2.0, 3.0}
        });

        var loss = Losses.mae().loss(predictions, targets);

        assertThat(loss).isCloseTo(0.0, within(EPSILON));
    }

}
