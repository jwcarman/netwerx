package org.jwcarman.netwerx.matrix;

@FunctionalInterface
public interface MatrixValueProvider {

// -------------------------- OTHER METHODS --------------------------

    double getValue(int row, int col);

}
