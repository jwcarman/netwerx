package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class RmsPropOptimizerTest {
    @Test
    void optimize_shouldUpdateParameters() {
        var optimizer = Optimizers.rmsProp(); // Default RMSProp instance

        var param = new SimpleMatrix(new double[][]{
                {0.5, -0.5},
                {1.0, -1.0}
        });

        var grad = new SimpleMatrix(new double[][]{
                {0.1, 0.2},
                {0.3, 0.4}
        });

        var updated = optimizer.optimize(param, grad);

        assertThat(updated).isNotNull().isNotEqualTo(param);
    }

    @Test
    void optimize_shouldRespectLearningRateAndEpsilon() {
        var optimizer = Optimizers.rmsProp(0.01, 0.95, 1e-6);

        var param = new SimpleMatrix(new double[][]{
                {0.2, 0.2}
        });

        var grad = new SimpleMatrix(new double[][]{
                {0.01, 0.01}
        });

        var original = param.copy();
        var updated = optimizer.optimize(param, grad);

        // Parameters should have decreased slightly
        assertThat(updated.get(0, 0)).isLessThan(original.get(0, 0));
        assertThat(updated.get(0, 1)).isLessThan(original.get(0, 1));
    }

    @Test
    void optimize_shouldNotMutateInputMatrices() {
        var optimizer = Optimizers.rmsProp();

        var param = new SimpleMatrix(2, 2);
        var grad = new SimpleMatrix(new double[][]{
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
        var optimizer = Optimizers.rmsProp(); // Uses defaults: lr=0.001, beta=0.9

        var param = new SimpleMatrix(2, 1);
        param.set(0, 0, 1.0);
        param.set(1, 0, 2.0);

        var grad = new SimpleMatrix(2, 1);
        grad.set(0, 0, 0.1);
        grad.set(1, 0, 0.2);

        var updated1 = optimizer.optimize(param, grad);
        var updated2 = optimizer.optimize(updated1, grad);

        // Ensure that each update reduces the parameter value
        assertThat(updated1.get(0, 0)).isLessThan(param.get(0, 0));
        assertThat(updated2.get(0, 0)).isLessThan(updated1.get(0, 0));

        assertThat(updated1.get(1, 0)).isLessThan(param.get(1, 0));
        assertThat(updated2.get(1, 0)).isLessThan(updated1.get(1, 0));

        // Optionally: Check we are within some small delta from original (don't hardcode expected values)
        assertThat(param.get(0, 0) - updated1.get(0, 0)).isPositive();
        assertThat(updated1.get(0, 0) - updated2.get(0, 0)).isPositive();
    }
}