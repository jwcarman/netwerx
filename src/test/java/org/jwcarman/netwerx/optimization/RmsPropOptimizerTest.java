package org.jwcarman.netwerx.optimization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;

class RmsPropOptimizerTest {
    @Test
    void optimize_shouldUpdateParameters() {
        var optimizer = Optimizers.<EjmlMatrix>rmsProp(); // Default RMSProp instance

        var param = Matrices.of(new double[][]{
                {0.5, -0.5},
                {1.0, -1.0}
        });

        var grad = Matrices.of(new double[][]{
                {0.1, 0.2},
                {0.3, 0.4}
        });

        var updated = optimizer.optimize(param, grad);

        assertThat(updated).isNotNull().isNotEqualTo(param);
    }

    @Test
    void optimize_shouldRespectLearningRateAndEpsilon() {
        var optimizer = Optimizers.<EjmlMatrix>rmsProp(0.01, 0.95, 1e-6);

        var param = Matrices.of(new double[][]{
                {0.2, 0.2}
        });

        var grad = Matrices.of(new double[][]{
                {0.01, 0.01}
        });

        var original = param.copy();
        var updated = optimizer.optimize(param, grad);

        // Parameters should have decreased slightly
        assertThat(updated.valueAt(0, 0)).isLessThan(original.valueAt(0, 0));
        assertThat(updated.valueAt(0, 1)).isLessThan(original.valueAt(0, 1));
    }

    @Test
    void optimize_shouldNotMutateInputMatrices() {
        var optimizer = Optimizers.<EjmlMatrix>rmsProp();

        var param = Matrices.of(2, 2);
        var grad = Matrices.of(new double[][]{
                {0.01, 0.02},
                {0.03, 0.04}
        });

        var paramCopy = param.copy();
        var gradCopy = grad.copy();

        optimizer.optimize(param, grad);

        assertThat(param.isIdentical(paramCopy, 1e-12)).isTrue();
        assertThat(grad.isIdentical(gradCopy, 1e-12)).isTrue();
    }

    @Test
    void optimize_shouldUpdateParametersAcrossCalls() {
        var optimizer = Optimizers.<EjmlMatrix>rmsProp(); // Uses defaults: lr=0.001, beta=0.9

        var param = Matrices.of(new double[][]{
                {1.0},
                {2.0}
        });
        var grad = Matrices.of(new double[][]{
                {0.1},
                {0.2}
        });

        var updated1 = optimizer.optimize(param, grad);
        var updated2 = optimizer.optimize(updated1, grad);

        // Ensure that each update reduces the parameter value
        assertThat(updated1.valueAt(0, 0)).isLessThan(param.valueAt(0, 0));
        assertThat(updated2.valueAt(0, 0)).isLessThan(updated1.valueAt(0, 0));

        assertThat(updated1.valueAt(1, 0)).isLessThan(param.valueAt(1, 0));
        assertThat(updated2.valueAt(1, 0)).isLessThan(updated1.valueAt(1, 0));

        // Optionally: Check we are within some small delta from original (don't hardcode expected values)
        assertThat(param.valueAt(0, 0) - updated1.valueAt(0, 0)).isPositive();
        assertThat(updated1.valueAt(0, 0) - updated2.valueAt(0, 0)).isPositive();
    }
}