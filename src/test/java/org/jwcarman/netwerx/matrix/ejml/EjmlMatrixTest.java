package org.jwcarman.netwerx.matrix.ejml;

import org.jwcarman.netwerx.matrix.AbstractMatrixTestCase;
import org.jwcarman.netwerx.matrix.MatrixFactory;

class EjmlMatrixTest extends AbstractMatrixTestCase<EjmlMatrix> {

// ------------------------------ FIELDS ------------------------------

    private final EjmlMatrixFactory factory = new EjmlMatrixFactory();

// -------------------------- OTHER METHODS --------------------------

    @Override
    protected MatrixFactory<EjmlMatrix> factory() {
        return factory;
    }

}
