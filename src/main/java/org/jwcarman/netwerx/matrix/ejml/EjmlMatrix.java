package org.jwcarman.netwerx.matrix.ejml;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixValueProvider;

public class EjmlMatrix implements Matrix<EjmlMatrix> {

// ------------------------------ FIELDS ------------------------------

    private final SimpleMatrix delegate;

// --------------------------- CONSTRUCTORS ---------------------------

    public EjmlMatrix(SimpleMatrix delegate) {
        this.delegate = delegate;
    }

// --------------------- GETTER / SETTER METHODS ---------------------

    public SimpleMatrix getDelegate() {
        return delegate;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Matrix ---------------------


    @Override
    public EjmlMatrix map(ElementMapper operation) {
        return new EjmlMatrix(delegate.elementOp(operation::apply));
    }

    @Override
    public double valueAt(int row, int column) {
        return delegate.get(row, column);
    }

    @Override
    public int rowCount() {
        return delegate.getNumRows();
    }

    @Override
    public int columnCount() {
        return delegate.getNumCols();
    }

    @Override
    public EjmlMatrix columnVector(int column) {
        return new EjmlMatrix(delegate.extractVector(false, column));
    }

    @Override
    public EjmlMatrix copy() {
        return new EjmlMatrix(delegate.copy());
    }

    @Override
    public EjmlMatrix fill(double value) {
        var copy = delegate.copy();
        copy.fill(value);
        return new EjmlMatrix(copy);
    }

    @Override
    public EjmlMatrix reshape(int rows, int columns) {
        var copy = delegate.copy();
        copy.reshape(rows, columns);
        return new EjmlMatrix(copy);
    }

    @Override
    public EjmlMatrix likeKind(int rows, int columns, MatrixValueProvider valueProvider) {
        var copy = new SimpleMatrix(rows, columns);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                copy.set(r, c, valueProvider.getValue(r, c));
            }
        }
        return new EjmlMatrix(copy);
    }

    @Override
    public EjmlMatrix multiply(EjmlMatrix other) {
        return new EjmlMatrix(delegate.mult(other.delegate));
    }

    @Override
    public EjmlMatrix rowVector(int row) {
        return new EjmlMatrix(delegate.extractVector(true, row));
    }

    @Override
    public EjmlMatrix transpose() {
        return new EjmlMatrix(delegate.transpose());
    }

}
