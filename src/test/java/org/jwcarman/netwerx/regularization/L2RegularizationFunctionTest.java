package org.jwcarman.netwerx.regularization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import static org.assertj.core.api.Assertions.assertThat;

class L2RegularizationFunctionTest {

    public static final double LAMBDA = 1e-3;

    @Test
    void testPenalty() {
        var factory = new EjmlMatrixFactory();
        RegularizationFunction<EjmlMatrix> reg = Regularizations.l2(LAMBDA);
        var matrix = factory.filled(2, 2, 1.0);

        double penalty = reg.penalty(matrix);
        assertThat(penalty).isEqualTo(matrix.sumOfSquares() * LAMBDA);
    }

    @Test
    void testGradient() {
        var factory = new EjmlMatrixFactory();
        RegularizationFunction<EjmlMatrix> reg = Regularizations.l2(LAMBDA);
        var matrix = factory.filled(2, 2, 1.0);

        var gradient = reg.gradient(matrix);

        assertThat(gradient).isNotNull();
        assertThat(gradient.rowCount()).isEqualTo(2);
        assertThat(gradient.columnCount()).isEqualTo(2);
        assertThat(gradient.elements().toList()).allMatch(e -> e.value() == matrix.valueAt(e.row(), e.column()) * 2 * LAMBDA);
    }
}