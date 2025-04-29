package org.jwcarman.netwerx.util;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;

import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Stream;

public class Matrices {

// -------------------------- STATIC METHODS --------------------------

    public static SimpleMatrix addColumnVector(SimpleMatrix a, SimpleMatrix columnVector) {
        final SimpleMatrix result = new SimpleMatrix(a.getNumRows(), a.getNumCols());
        for (int row = 0; row < a.getNumRows(); row++) {
            for (int col = 0; col < a.getNumCols(); col++) {
                result.set(row, col, a.get(row, col) + columnVector.get(row));
            }
        }
        return result;
    }

    public static SimpleMatrix subtractColumnVector(SimpleMatrix a, SimpleMatrix columnVector) {
        final SimpleMatrix result = new SimpleMatrix(a.getNumRows(), a.getNumCols());
        for (int row = 0; row < a.getNumRows(); row++) {
            for (int col = 0; col < a.getNumCols(); col++) {
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
        return original.elementOp((SimpleOperations.ElementOpReal) (_, _, value) -> fn.applyAsDouble(value));
    }

    public static Stream<PredictionTarget> predictionTargets(SimpleMatrix predictions, SimpleMatrix targets) {
        if (predictions.getNumRows() != targets.getNumRows() || predictions.getNumCols() != targets.getNumCols()) {
            throw new IllegalArgumentException("Predictions and targets must have the same dimensions.");
        }
        return Streams.zip(
                allElements(predictions),
                allElements(targets),
                (pred, target) -> new PredictionTarget(
                        pred.row(),
                        pred.col(),
                        pred.value(),
                        target.value()
                )
        );
    }

    public static Stream<MatrixElement> allElements(SimpleMatrix matrix) {
        return Stream.generate(new ElementSupplier(matrix))
                .limit((long) matrix.getNumRows() * matrix.getNumCols());
    }

    public static Stream<MatrixElement> rowElements(SimpleMatrix matrix, final int row) {
        if (row < 0 || row >= matrix.getNumRows()) {
            throw new IndexOutOfBoundsException("Row index out of bounds: " + row);
        }
        return Stream.iterate(0, col -> col + 1)
                .limit(matrix.getNumCols())
                .map(col -> new MatrixElement(row, col, matrix.get(row, col)));
    }

    public static Stream<MatrixElement> columnElements(SimpleMatrix matrix, final int col) {
        if (col < 0 || col >= matrix.getNumCols()) {
            throw new IndexOutOfBoundsException("Column index out of bounds: " + col);
        }
        return Stream.iterate(0, row -> row + 1)
                .limit(matrix.getNumRows())
                .map(row -> new MatrixElement(row, col, matrix.get(row, col)));
    }

    public static void forEachElement(SimpleMatrix matrix, MatrixElementConsumer consumer) {
        for (int row = 0; row < matrix.getNumRows(); row++) {
            for (int col = 0; col < matrix.getNumCols(); col++) {
                consumer.accept(new MatrixElement(row, col, matrix.get(row, col)));
            }
        }
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Matrices() {

    }

// -------------------------- INNER CLASSES --------------------------

    public record PredictionTarget(int row, int col, double prediction, double target) {

    }

    @FunctionalInterface
    public interface MatrixElementConsumer {

// -------------------------- OTHER METHODS --------------------------

        void accept(MatrixElement element);

    }

    public record MatrixElement(int row, int col, double value) {

    }

    private static class ElementSupplier implements Supplier<MatrixElement> {

// ------------------------------ FIELDS ------------------------------

        private final SimpleMatrix matrix;
        private int row = 0;
        private int col = 0;

// --------------------------- CONSTRUCTORS ---------------------------

        ElementSupplier(SimpleMatrix matrix) {
            this.matrix = matrix;
        }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Supplier ---------------------

        @Override
        public MatrixElement get() {
            final var element = new MatrixElement(row, col, matrix.get(row, col));
            col++;
            if (col >= matrix.getNumCols()) {
                col = 0;
                row++;
            }
            return element;
        }

    }

}
