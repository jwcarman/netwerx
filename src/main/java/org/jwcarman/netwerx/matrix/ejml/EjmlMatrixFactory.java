package org.jwcarman.netwerx.matrix.ejml;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;

public class EjmlMatrixFactory implements MatrixFactory<EjmlMatrix> {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MatrixFactory ---------------------
    private static final EjmlMatrix EMPTY = new EjmlMatrix(new SimpleMatrix(0, 0));

    @Override
    public EjmlMatrix empty() {
        return EMPTY;
    }

    @Override
    public EjmlMatrix from(double[][] data) {
        return new EjmlMatrix(new SimpleMatrix(data));
    }

    @Override
    public EjmlMatrix zeros(int rows, int columns) {
        return new EjmlMatrix(new SimpleMatrix(rows, columns));
    }

    @Override
    public EjmlMatrix filled(int rows, int columns, MatrixValueSupplier values) {
        var matrix = new SimpleMatrix(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                matrix.set(row, col, values.getValue(row, col));
            }
        }
        return new EjmlMatrix(matrix);
    }

    @Override
    public EjmlMatrix filled(int rows, int columns, double constantValue) {
        var matrix = new SimpleMatrix(rows, columns);
        matrix.fill(constantValue);
        return new EjmlMatrix(matrix);
    }

}
