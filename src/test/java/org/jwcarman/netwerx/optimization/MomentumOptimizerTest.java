package org.jwcarman.netwerx.optimization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.learning.LearningRateProviders;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class MomentumOptimizerTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void optimize_shouldAccumulateMomentum_acrossCalls() {
        var optimizer = Optimizers.<EjmlMatrix>momentum(0.01, 0.9);

        var params = Matrices.of(new double[][]{
                {1.0},
                {2.0}
        });

        var grad = Matrices.of(new double[][]{
                {0.1},
                {0.2}
        });

        // First update
        var updated1 = optimizer.optimize(10, params, grad);

        // Second update with same gradient
        var updated2 = optimizer.optimize(10, updated1, grad);

        // Velocity should now be:
        // v1 = 0.01 * grad
        // v2 = 0.9 * v1 + 0.01 * grad = 0.009 * grad + 0.01 * grad = 0.019 * grad
        // So update should be bigger than the first one
        double expectedV2 = 0.019 * 0.1;
        assertThat(updated2.valueAt(0, 0)).isCloseTo(1.0 - 0.001 - expectedV2, withinTolerance());
    }

    @Test
    void optimize_shouldAccumulateMomentum_acrossCallsWithLearningRateProvider() {
        var optimizer = Optimizers.<EjmlMatrix>momentum(LearningRateProviders.constant(0.01), 0.9);

        var params = Matrices.of(new double[][]{
                {1.0},
                {2.0}
        });

        var grad = Matrices.of(new double[][]{
                {0.1},
                {0.2}
        });

        // First update
        var updated1 = optimizer.optimize(10, params, grad);

        // Second update with same gradient
        var updated2 = optimizer.optimize(10, updated1, grad);

        // Velocity should now be:
        // v1 = 0.01 * grad
        // v2 = 0.9 * v1 + 0.01 * grad = 0.009 * grad + 0.01 * grad = 0.019 * grad
        // So update should be bigger than the first one
        double expectedV2 = 0.019 * 0.1;
        assertThat(updated2.valueAt(0, 0)).isCloseTo(1.0 - 0.001 - expectedV2, withinTolerance());
    }

    @Test
    void optimize_shouldUpdateParameters_withMomentum() {
        var optimizer = Optimizers.<EjmlMatrix>momentum(); // default: 0.01 learning rate, 0.9 momentum

        var params = Matrices.of(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });

        var grad = Matrices.of(new double[][]{
                {0.1, 0.1},
                {0.2, 0.2}
        });

        var updated = optimizer.optimize(7, params, grad);

        // velocity = 0.9 * 0 + 0.01 * grad = 0.001 * grad
        // expected update = param - velocity
        assertThat(updated.valueAt(0, 0)).isCloseTo(1.0 - 0.001, withinTolerance());
        assertThat(updated.valueAt(1, 1)).isCloseTo(4.0 - 0.002, withinTolerance());
    }

}
