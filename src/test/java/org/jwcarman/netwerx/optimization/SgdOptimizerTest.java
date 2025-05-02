package org.jwcarman.netwerx.optimization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class SgdOptimizerTest {
    @Test
    void optimize_shouldUpdateParameters() {
        var optimizer = Optimizers.<EjmlMatrix>sgd(); // Default learning rate (0.01)

        var param = Matrices.of(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });

        var grad = Matrices.of(new double[][]{
                {0.1, 0.1},
                {0.2, 0.2}
        });

        var updated = optimizer.optimize(param, grad);
        assertThat(updated.valueAt(0, 0)).isCloseTo(1.0 - 0.001, withinTolerance());
        assertThat(updated.valueAt(1, 0)).isCloseTo(2.998, withinTolerance());
    }

    @Test
    void optimize_shouldNotMutateInputMatrices() {
        var optimizer = Optimizers.<EjmlMatrix>sgd();

        var param = Matrices.of(new double[][]{{1.0, 2.0}});
        var grad = Matrices.of(new double[][]{{0.1, 0.2}});

        var paramCopy = param.copy();
        var gradCopy = grad.copy();

        optimizer.optimize(param, grad);

        assertThat(param.isIdentical(paramCopy, 1e-12)).isTrue();
        assertThat(grad.isIdentical(gradCopy, 1e-12)).isTrue();
    }

    @Test
    void optimize_shouldRespectCustomLearningRate() {
        var optimizer = Optimizers.<EjmlMatrix>sgd(0.5);

        var param = Matrices.of(new double[][]{{10.0}});
        var grad = Matrices.of(new double[][]{{2.0}});

        var updated = optimizer.optimize(param, grad);

        assertThat(updated.valueAt(0, 0)).isCloseTo(9.0, withinTolerance());
    }
}