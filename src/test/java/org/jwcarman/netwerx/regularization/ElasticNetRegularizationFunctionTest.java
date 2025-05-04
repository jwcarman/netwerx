package org.jwcarman.netwerx.regularization;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import static org.assertj.core.api.Assertions.assertThat;

class ElasticNetRegularizationFunctionTest {
    public static final double L1 = 1e-3;
    public static final double L2 = 1e-4;

    @Test
    void testPenalty() {
        var factory = new EjmlMatrixFactory();
        RegularizationFunction<EjmlMatrix> reg = Regularizations.elasticNet(L1, L2);
        var matrix = factory.filled(2, 2, 1.0);

        double penalty = reg.penalty(matrix);
        assertThat(penalty).isEqualTo(matrix.sumOfSquares() * L1 + matrix.sumOfAbs() * L2);
    }

    @Test
    void testGradient() {
        var factory = new EjmlMatrixFactory();
        RegularizationFunction<EjmlMatrix> reg = Regularizations.elasticNet(L1, L2);
        var matrix = factory.filled(2, 2, 1.0);

        var gradient = reg.gradient(matrix);

        assertThat(gradient).isNotNull();
        assertThat(gradient.rowCount()).isEqualTo(2);
        assertThat(gradient.columnCount()).isEqualTo(2);
        assertThat(gradient.elements().toList()).allMatch(e -> e.value() == (L1 * Math.signum(matrix.valueAt(e.row(), e.column()))) + (L2 * 2 * matrix.valueAt(e.row(), e.column())));
    }

}