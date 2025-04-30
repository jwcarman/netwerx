package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class SgdOptimizerTest {
    @Test
    void optimize_shouldUpdateParameters() {
        var optimizer = Optimizers.sgd(); // Default learning rate (0.01)

        var param = new SimpleMatrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });

        var grad = new SimpleMatrix(new double[][]{
                {0.1, 0.1},
                {0.2, 0.2}
        });

        var updated = optimizer.optimize(param, grad);
        assertThat(updated.get(0, 0)).isCloseTo(1.0 - 0.001, withinTolerance());
        assertThat(updated.get(1, 0)).isCloseTo(2.998, withinTolerance());
    }

    @Test
    void optimize_shouldNotMutateInputMatrices() {
        var optimizer = Optimizers.sgd();

        var param = new SimpleMatrix(new double[][]{{1.0, 2.0}});
        var grad = new SimpleMatrix(new double[][]{{0.1, 0.2}});

        var paramCopy = param.copy();
        var gradCopy = grad.copy();

        optimizer.optimize(param, grad);

        assertThat(param.isIdentical(paramCopy, 1e-12)).isTrue();
        assertThat(grad.isIdentical(gradCopy, 1e-12)).isTrue();
    }

    @Test
    void optimize_shouldRespectCustomLearningRate() {
        var optimizer = Optimizers.sgd(0.5);

        var param = new SimpleMatrix(new double[][]{{10.0}});
        var grad = new SimpleMatrix(new double[][]{{2.0}});

        var updated = optimizer.optimize(param, grad);

        assertThat(updated.get(0, 0)).isCloseTo(9.0, withinTolerance());
    }
}