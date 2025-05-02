package org.jwcarman.netwerx.matrix.ejml;

import org.jwcarman.netwerx.matrix.AbstractMatrixFactoryTestCase;

class EjmlMatrixFactoryTest extends AbstractMatrixFactoryTestCase<EjmlMatrix> {

// ------------------------------ FIELDS ------------------------------

    private final EjmlMatrixFactory factory = new EjmlMatrixFactory();

// -------------------------- OTHER METHODS --------------------------

    @Override
    protected EjmlMatrixFactory factory() {
        return factory;
    }

}
