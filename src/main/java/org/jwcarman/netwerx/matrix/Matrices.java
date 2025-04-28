package org.jwcarman.netwerx.matrix;

import org.ejml.simple.SimpleMatrix;

import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;

public class Matrices {

// -------------------------- STATIC METHODS --------------------------

    public static SimpleMatrix addColumnVector(SimpleMatrix a, SimpleMatrix columnVector) {
        final SimpleMatrix result = new SimpleMatrix(a.getNumRows(), a.getNumCols());
        for(int row = 0; row < a.getNumRows(); row++) {
            for(int col = 0; col < a.getNumCols(); col++) {
                result.set(row, col, a.get(row, col) + columnVector.get(row));
            }
        }
        return result;
    }

    public static SimpleMatrix subtractColumnVector(SimpleMatrix a, SimpleMatrix columnVector) {
        final SimpleMatrix result = new SimpleMatrix(a.getNumRows(), a.getNumCols());
        for(int row = 0; row < a.getNumRows(); row++) {
            for(int col = 0; col < a.getNumCols(); col++) {
                result.set(row, col, a.get(row, col) - columnVector.get(row));
            }
        }
        return result;
    }

    public static SimpleMatrix filled(int rows, int cols, DoubleSupplier filler) {
        final SimpleMatrix result = new SimpleMatrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.set(row, col, filler.getAsDouble());
            }
        }
        return result;
    }
    public static SimpleMatrix clamp(SimpleMatrix original, double min, double max) {
        return apply(original, value -> Math.clamp(value, min, max));
    }

    public static SimpleMatrix apply(SimpleMatrix original, DoubleUnaryOperator fn) {
        final SimpleMatrix result = new SimpleMatrix(original);
        result.elementOp((row, col, value) -> {
            return fn.applyAsDouble(value);
        });
        return result;
    }

}
