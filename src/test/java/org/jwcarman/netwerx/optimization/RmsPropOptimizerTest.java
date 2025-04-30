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
}