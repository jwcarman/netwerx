package org.jwcarman.netwerx.matrix.ejml;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.AbstractMatrixTestCase;
import org.jwcarman.netwerx.matrix.MatrixFactory;

import static org.assertj.core.api.Assertions.assertThat;

class EjmlMatrixTest extends AbstractMatrixTestCase<EjmlMatrix> {

// ------------------------------ FIELDS ------------------------------

    private final EjmlMatrixFactory factory = new EjmlMatrixFactory();

// -------------------------- OTHER METHODS --------------------------

    @Override
    protected MatrixFactory<EjmlMatrix> factory() {
        return factory;
    }

    @Test
    void testDelegate() {
        SimpleMatrix delegate = new SimpleMatrix(2, 2);
        var matrix = new EjmlMatrix(delegate);
        assertThat(matrix.getDelegate()).isSameAs(delegate);
    }

}
