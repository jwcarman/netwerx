package org.jwcarman.netwerx.matrix.ejml;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.function.ToDoubleFunction;

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
    public EjmlMatrix add(EjmlMatrix other) {
        return new EjmlMatrix(delegate.plus(other.delegate));
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
    public EjmlMatrix elementMultiply(double scalar) {
        return new EjmlMatrix(delegate.scale(scalar));
    }

    @Override
    public EjmlMatrix map(ElementOperation operation) {
        return new EjmlMatrix(delegate.elementOp(operation::apply));
    }

    @Override
    public EjmlMatrix fill(double value) {
        var copy = delegate.copy();
        copy.fill(value);
        return new EjmlMatrix(copy);
    }

    @Override
    public EjmlMatrix elementMultiply(EjmlMatrix other) {
        return new EjmlMatrix(delegate.elementMult(other.delegate));
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
    public double maxAbs() {
        return delegate.elementMaxAbs();
    }

    @Override
    public EjmlMatrix reshape(int rows, int columns) {
        var copy = delegate.copy();
        copy.reshape(rows, columns);
        return new EjmlMatrix(copy);
    }

    @Override
    public double sum() {
        return delegate.elementSum();
    }

    @Override
    public EjmlMatrix multiply(EjmlMatrix other) {
        return new EjmlMatrix(delegate.mult(other.delegate));
    }

    @Override
    public EjmlMatrix negate() {
        return new EjmlMatrix(delegate.negative());
    }

    @Override
    public EjmlMatrix rowVector(int row) {
        return new EjmlMatrix(delegate.extractVector(true, row));
    }

    @Override
    public EjmlMatrix subtract(EjmlMatrix other) {
        return new EjmlMatrix(delegate.minus(other.delegate));
    }

    @Override
    public EjmlMatrix reduceRows(ToDoubleFunction<EjmlMatrix> reducer) {
        SimpleMatrix agg = new SimpleMatrix(delegate.getNumRows(), 1);
        for (int row = 0; row < delegate.getNumRows(); row++) {
            agg.set(row, 0, reducer.applyAsDouble(rowVector(row)));
        }
        return new EjmlMatrix(agg);
    }

    @Override
    public EjmlMatrix reduceColumns(ToDoubleFunction<EjmlMatrix> reducer) {
        SimpleMatrix agg = new SimpleMatrix(1, delegate.getNumCols());
        for (int col = 0; col < delegate.getNumCols(); col++) {
            agg.set(0, col, reducer.applyAsDouble(columnVector(col)));
        }
        return new EjmlMatrix(agg);
    }

    @Override
    public EjmlMatrix transpose() {
        return new EjmlMatrix(delegate.transpose());
    }

    @Override
    public double valueAt(int row, int column) {
        return delegate.get(row, column);
    }



    @Override
    public EjmlMatrix likeKind(int rows, int columns) {
        return new EjmlMatrix(new SimpleMatrix(rows, columns));
    }

}
