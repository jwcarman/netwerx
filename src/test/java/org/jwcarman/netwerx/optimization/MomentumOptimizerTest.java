package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class MomentumOptimizerTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void optimize_shouldAccumulateMomentum_acrossCalls() {
        var optimizer = Optimizers.momentum(0.01, 0.9);

        var params = new SimpleMatrix(new double[][]{
                {1.0},
                {2.0}
        });

        var grad = new SimpleMatrix(new double[][]{
                {0.1},
                {0.2}
        });

        // First update
        var updated1 = optimizer.optimize(params, grad);

        // Second update with same gradient
        var updated2 = optimizer.optimize(updated1, grad);

        // Velocity should now be:
        // v1 = 0.01 * grad
        // v2 = 0.9 * v1 + 0.01 * grad = 0.009 * grad + 0.01 * grad = 0.019 * grad
        // So update should be bigger than the first one
        double expectedV2 = 0.019 * 0.1;
        assertThat(updated2.get(0, 0)).isCloseTo(1.0 - 0.001 - expectedV2, withinTolerance());
    }

    @Test
    void optimize_shouldUpdateParameters_withMomentum() {
        var optimizer = Optimizers.momentum(); // default: 0.01 learning rate, 0.9 momentum

        var params = new SimpleMatrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });

        var grad = new SimpleMatrix(new double[][]{
                {0.1, 0.1},
                {0.2, 0.2}
        });

        var updated = optimizer.optimize(params, grad);

        // velocity = 0.9 * 0 + 0.01 * grad = 0.001 * grad
        // expected update = param - velocity
        assertThat(updated.get(0, 0)).isCloseTo(1.0 - 0.001, withinTolerance());
        assertThat(updated.get(1, 1)).isCloseTo(4.0 - 0.002, withinTolerance());
    }

}
