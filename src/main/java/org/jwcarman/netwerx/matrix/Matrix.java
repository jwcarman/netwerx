package org.jwcarman.netwerx.matrix;

import java.util.Comparator;
import java.util.NoSuchElementException;
import java.util.OptionalDouble;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public interface Matrix<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Adds another matrix to this matrix.
     *
     * @param other the matrix to add
     * @return a new matrix that is the result of the addition
     */
    M add(M other);

    /**
     * Adds a column vector to each column of the matrix.
     *
     * @param columnVector the column vector to add
     * @return a new matrix with the column vector added to each column
     */
    default M addColumnVector(M columnVector) {
        return map((row, col, value) -> value + columnVector.valueAt(row, 0));
    }

    /**
     * Applies a function to each element of the matrix.
     *
     * @param operation the operation to apply
     * @return a new matrix with the operation applied to each element
     */
    M map(ElementOperation operation);

    double valueAt(int row, int column);

    /**
     * Adds a row vector to each row of the matrix.
     *
     * @param rowVector the row vector to add
     * @return a new matrix with the row vector added to each row
     */
    default M addRowVector(M rowVector) {
        return map((row, col, value) -> value + rowVector.valueAt(0, col));
    }

    default M clamp(double min, double max) {
        return map((_, _, value) -> Math.clamp(value, min, max));
    }

    default double columnMax(int column) {
        return columnValues(column).max().orElseThrow(NoSuchElementException::new);
    }

    default M columnMean() {
        return reduceColumns(m -> m.sum() / m.rowCount());
    }

    M reduceColumns(ToDoubleFunction<M> reducer);

    /**
     * Returns the sum of all elements in the matrix.
     *
     * @return the sum of all elements
     */
    double sum();

    /**
     * Returns the number of rows in the matrix.
     *
     * @return the number of rows
     */
    int rowCount();

    default double columnMean(int column) {
        return columnValues(column).average().orElseThrow(NoSuchElementException::new);
    }

    default M columnMin() {
        return reduceColumns(Matrix::min);
    }

    default double columnMin(int column) {
        return columnValues(column).min().orElseThrow(NoSuchElementException::new);
    }

    default M columnStd() {
        return reduceColumns(col -> Math.sqrt(col.variance()));
    }

    default double variance() {
        return variance(this::values);
    }

    default double columnStd(int column) {
        return Math.sqrt(columnVariance(column));
    }

    default double columnVariance(int column) {
        return variance(() -> columnValues(column));
    }

    private double variance(Supplier<DoubleStream> stream) {
        var mean = stream.get().average().orElse(0.0);
        return stream.get()
                .map(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);
    }

    default DoubleStream columnValues(int column) {
        return columnVector(column).values();
    }

    default DoubleStream values() {
        return IntStream.range(0, size())
                .mapToDouble(index -> valueAt(index / columnCount(), index % columnCount()));
    }

    /**
     * Returns the number of columns in the matrix.
     *
     * @return the number of columns
     */
    int columnCount();

    default double columnSum(int column) {
        return columnVector(column).sum();
    }

    default M columnVariance() {
        return reduceColumns(col -> variance(col::values));
    }

    /**
     * Returns the specified column as a column vector.
     *
     * @param column the index of the column to extract
     * @return the column vector
     */
    M columnVector(int column);


    /**
     * Returns a new matrix that is a copy of this matrix.
     *
     * @return a new matrix that is a copy of this matrix
     */
    M copy();

    /**
     * Returns a new matrix with each element replaced by its absolute value.
     *
     * @return a new matrix with each element replaced by its absolute value
     */
    default M elementAbs() {
        return map((_, _, value) -> Math.abs(value));
    }

    default M elementAdd(double value) {
        return map((_, _, v) -> v + value);
    }

    default M elementDivide(double value) {
        return map((_, _, v) -> v / value);
    }

    default M elementDivide(M other) {
        return map((row, col, value) -> value / other.valueAt(row, col));
    }

    /**
     * Multiplies the matrix by another matrix element-wise (hadamard product).
     *
     * @param other the matrix to multiply by
     * @return a new matrix that is the result of the element-wise multiplication (hadamard product)
     */
    M elementMultiply(M other);

    default M elementSqrt() {
        return map((_, _, v) -> Math.sqrt(v));
    }

    default M elementSquare() {
        return elementPower(2.0);
    }

    /**
     * Returns a new matrix with each element raised to the specified power.
     *
     * @param exponent the exponent to raise each element to
     * @return a new matrix with each element raised to the specified power
     */
    default M elementPower(double exponent) {
        return map((_, _, value) -> Math.pow(value, exponent));
    }

    default M elementSubtract(double value) {
        return map((_, _, v) -> v - value);
    }

    default Stream<MatrixElement> elements() {
        return IntStream.range(0, size())
                .mapToObj(index -> {
                    var row = index / columnCount();
                    var column = index % columnCount();
                    return new MatrixElement(row, column, valueAt(row, column));
                });
    }

    M fill(double value);

    /**
     * Returns a new 1-column matrix containing all elements in row-major order.
     * This is equivalent to flattening the matrix into an N × 1 shape.
     *
     * @return a new matrix of shape (size(), 1)
     */
    default M flattenToColumn() {
        return reshape(size(), 1);
    }

    /**
     * Returns a new matrix with the specified number of rows and columns.
     * The elements are filled in row-major order.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @return a new matrix with the specified shape
     */
    M reshape(int rows, int columns);

    /**
     * Returns the size of the matrix (number of elements).
     *
     * @return the size of the matrix
     */
    default int size() {
        return rowCount() * columnCount();
    }

    /**
     * Returns a new 1-row matrix containing all elements in row-major order.
     * This is equivalent to flattening the matrix into a 1 × N shape.
     *
     * @return a new matrix of shape (1, size())
     */
    default M flattenToRow() {
        return reshape(1, size());
    }

    default boolean isIdentical(M other, double tolerance) {
        return elements().allMatch(e -> {
            var otherValue = other.valueAt(e.row(), e.column());
            return Math.abs(e.value() - otherValue) <= tolerance;
        });
    }

    /**
     * Checks if the matrix is square (number of rows equals number of columns).
     *
     * @return true if the matrix is square, false otherwise
     */
    default boolean isSquare() {
        return rowCount() == columnCount();
    }

    M likeKind(int rows, int columns);

    default M log() {
        return map((_, _, value) -> Math.log(value));
    }

    default double max() {
        return deref(values().max());
    }

    private static double deref(OptionalDouble optional) {
        return optional.orElseThrow(NoSuchElementException::new);
    }

    double maxAbs();

    /**
     * Returns the mean of all elements in the matrix.
     *
     * @return the mean of all elements
     */
    default double mean() {
        return sum() / size();
    }

    default double min() {
        return deref(values().min());
    }

    /**
     * Multiplies the matrix by another matrix.
     *
     * @param other the matrix to multiply by
     * @return a new matrix that is the result of the multiplication
     */
    M multiply(M other);

    /**
     * Negates the matrix.
     *
     * @return a new matrix that is the negation of the original
     */
    M negate();

    default double normL1() {
        return values().map(Math::abs).sum();
    }

    default double normL2() {
        return Math.sqrt(values().map(v -> v * v).sum());
    }

    default int rowArgMax(int row) {
        var columnCount = columnCount();
        return IntStream.range(0, columnCount)
                .boxed()
                .max(Comparator.comparingDouble(col -> valueAt(row, col)))
                .orElseThrow(NoSuchElementException::new);
    }

    default M rowMax() {
        return reduceRows(Matrix::max);
    }

    M reduceRows(ToDoubleFunction<M> reducer);

    default double rowMax(int row) {
        return deref(rowValues(row).max());
    }

    default M rowMean() {
        return reduceRows(m -> m.sum() / m.columnCount());
    }

    default double rowMean(int row) {
        return deref(rowValues(row).average());
    }

    default M rowMin() {
        return reduceRows(Matrix::min);
    }

    default double rowMin(int row) {
        return deref(rowValues(row).min());
    }

    default M rowStd() {
        return reduceRows(row -> Math.sqrt(row.variance()));
    }

    default double rowStd(int row) {
        return Math.sqrt(rowVariance(row));
    }

    default double rowVariance(int row) {
        return variance(() -> rowValues(row));
    }

    default DoubleStream rowValues(int row) {
        return rowVector(row).values();
    }

    /**
     * Sums the rows of the matrix.
     *
     * @return a new column vector with the sum of each row
     */
    default M rowSum() {
        return reduceRows(Matrix::sum);
    }

    default double rowSum(int row) {
        return rowVector(row).sum();
    }

    default M rowVariance() {
        return reduceRows(row -> variance(row::values));
    }

    /**
     * Returns the specified row as a row vector.
     *
     * @param row the index of the row to extract
     * @return the row vector
     */
    M rowVector(int row);

    /**
     * Multiplies all elements of the matrix by a scalar.
     * <p>
     * This method is an alias for {@link #elementMultiply(double)}.
     *
     * @param scalar the scalar value to multiply by
     * @return a new matrix with each element multiplied by the scalar
     * @see #elementMultiply(double)
     */
    default M scale(double scalar) {
        return elementMultiply(scalar);
    }

    /**
     * Returns a new matrix with each element multiplied by the scalar.
     *
     * @param scalar the scalar value to multiply by
     * @return a new matrix with each element multiplied by the scalar
     */
    M elementMultiply(double scalar);

    default M softmax() {
        M maxPerColumn = this.columnMax(); // shape: (1, columns)
        M stabilized = this.subtractRowVector(maxPerColumn);
        M exp = stabilized.exp();
        M sumPerColumn = exp.columnSum(); // shape: (1, columns)

        return exp.map((row, col, value) -> value / sumPerColumn.valueAt(0, col));
    }

    default M columnMax() {
        return reduceColumns(Matrix::max);
    }

    default M subtractRowVector(M rowVector) {
        return map((row, col, value) -> value - rowVector.valueAt(0, col));
    }

    /**
     * Returns a new matrix with each element replaced by e^x.
     *
     * @return a new matrix with each element replaced by e^x
     */
    default M exp() {
        return map((_, _, value) -> Math.exp(value));
    }

    default M columnSum() {
        return reduceColumns(Matrix::sum);
    }

    default double std() {
        return Math.sqrt(variance());
    }

    M subtract(M other);

    default M subtractColumnVector(M columnVector) {
        return map((row, col, value) -> value - columnVector.valueAt(row, 0));
    }

    /**
     * Returns the transpose of the matrix.
     *
     * @return the transposed matrix as a new instance
     */
    M transpose();

// -------------------------- INNER CLASSES --------------------------

    @FunctionalInterface
    interface ElementOperation {

// -------------------------- OTHER METHODS --------------------------

        double apply(int row, int column, double value);

    }

    record MatrixElement(int row, int column, double value) {

    }

}
