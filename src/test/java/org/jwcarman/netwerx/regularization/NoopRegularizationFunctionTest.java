package org.jwcarman.netwerx.regularization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import static org.assertj.core.api.Assertions.assertThat;

class NoopRegularizationFunctionTest {

    @Test
    void testGradient() {
        var factory = new EjmlMatrixFactory();
        var noop = new NoopRegularizationFunction<EjmlMatrix>();
        var matrix = factory.filled(2, 2, 1.0);
        var gradient = noop.gradient(matrix);
        assertThat(gradient).isNotNull();
        assertThat(gradient.rowCount()).isEqualTo(2);
        assertThat(gradient.columnCount()).isEqualTo(2);
        assertThat(gradient.values().boxed().allMatch(v -> v == 0.0)).isTrue();
    }

    @Test
    void testPenalty() {
        var factory = new EjmlMatrixFactory();
        var noop = new NoopRegularizationFunction<EjmlMatrix>();
        var matrix = factory.filled(2, 2, 1.0);
        double penalty = noop.penalty(matrix);
        assertThat(penalty).isEqualTo(0.0);
    }

}