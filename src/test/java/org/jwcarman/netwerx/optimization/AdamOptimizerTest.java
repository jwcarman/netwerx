package org.jwcarman.netwerx.optimization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;

class AdamOptimizerTest {

    @Test
    void optimize_shouldUpdateParameters() {
        var param = Matrices.of(2, 2);
        var grad = Matrices.of(new double[][]{
                {0.1, 0.2},
                {0.3, 0.4}
        });

        var optimizer = Optimizers.<EjmlMatrix>adam(); // Default constructor

        var updated = optimizer.optimize(param, grad);

        assertThat(updated).isNotEqualTo(param);
    }

    @Test
    void constructor_withCustomHyperparameters_shouldRespectValues() {
        double learningRate = 0.005;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = 1e-6;

        var optimizer = Optimizers.<EjmlMatrix>adam(learningRate, beta1, beta2, epsilon);

        var param = Matrices.of(2, 2);
        var grad = Matrices.of(new double[][]{
                {0.01, 0.02},
                {0.03, 0.04}
        });

        var paramCopy = param.copy();
        var gradCopy = grad.copy();

        var updated = optimizer.optimize(param, grad);

        assertThat(updated).isNotNull();
        assertThat(param.isIdentical(paramCopy, 1e-12)).isTrue();
        assertThat(grad.isIdentical(gradCopy, 1e-12)).isTrue();
    }

    @Test
    void optimize_shouldConvergeWithSmallGradients() {
        var optimizer = Optimizers.<EjmlMatrix>adam(0.001, 0.9, 0.999, 1e-8);
        var param = Matrices.of(2, 1);
        var grad = Matrices.of(new double[][]{{1e-5}, {1e-5}});

        var original = param.copy();
        var updated = optimizer.optimize(param, grad);

        // Should nudge the parameter slightly
        assertThat(updated.valueAt(0, 0)).isNotEqualTo(original.valueAt(0, 0));
    }

    @Test
    void optimizer_shouldUpdateParametersUsingAdam_twice() {
        var param = Matrices.of(new double[][] {
                {1.0},
                {2.0}
        });
        var grad = Matrices.of(new double[][] {
                {0.1},
                {0.2}
        });

        var optimizer = Optimizers.<EjmlMatrix>adam(); // default config

        // First step (t = 1)
        var updated1 = optimizer.optimize(param, grad);
        // Second step (t = 2)
        var updated2 = optimizer.optimize(updated1, grad);

        // Assert directional correctness
        assertThat(updated1.valueAt(0, 0)).isLessThan(1.0);
        assertThat(updated1.valueAt(1, 0)).isLessThan(2.0);

        assertThat(updated2.valueAt(0, 0)).isLessThan(updated1.valueAt(0, 0));
        assertThat(updated2.valueAt(1, 0)).isLessThan(updated1.valueAt(1, 0));

        // Assert reasonable values (sanity checks)
        assertThat(updated2.valueAt(0, 0)).isBetween(0.995, 0.999);
        assertThat(updated2.valueAt(1, 0)).isBetween(1.995, 1.999);
    }
}