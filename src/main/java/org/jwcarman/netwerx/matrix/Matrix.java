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
    default M add(M other) {
        return map((row, col, value) -> value + other.valueAt(row, col));
    }

    /**
     * Applies a function to each element of the matrix.
     *
     * @param operation the operation to apply
     * @return a new matrix with the operation applied to each element
     */
    M map(ElementOperation operation);

    /**
     * Returns the value at the specified row and column.
     *
     * @param row    the index of the row
     * @param column the index of the column
     * @return the value at the specified row and column
     */
    double valueAt(int row, int column);

    /**
     * Adds a column vector to each column of the matrix.
     *
     * @param columnVector the column vector to add
     * @return a new matrix with the column vector added to each column
     */
    default M addColumnVector(M columnVector) {
        return map((row, _, value) -> value + columnVector.valueAt(row, 0));
    }

    /**
     * Adds a row vector to each row of the matrix.
     *
     * @param rowVector the row vector to add
     * @return a new matrix with the row vector added to each row
     */
    default M addRowVector(M rowVector) {
        return map((_, col, value) -> value + rowVector.valueAt(0, col));
    }

    default M binaryClassifierOutputs(boolean[] labels) {
        if (labels.length != columnCount()) {
            throw new IllegalArgumentException(String.format("Label count %d must match column count %d", labels.length, columnCount()));
        }
        return likeKind(1, labels.length)
                .map((_, col, _) -> labels[col] ? 1.0 : 0.0);
    }

    /**
     * Returns a new matrix with each element clamped to the specified range.
     *
     * @param min the minimum value
     * @param max the maximum value
     * @return a new matrix with each element clamped to the specified range
     * @see Math#clamp(double, double, double)
     */
    default M clamp(double min, double max) {
        return map((_, _, value) -> Math.clamp(value, min, max));
    }

    /**
     * Returns a row vector containing the mean of each column.
     *
     * @return a new row vector with the mean of each column
     */
    default M columnMean() {
        return reduceColumns(m -> m.sum() / m.rowCount());
    }

    /**
     * Returns a new row vector containing the values provided by the supplied reducer function.
     *
     * @param reducer the reducer function to apply to each column
     * @return a row vector with the values provided by the reducer
     */
    default M reduceColumns(ToDoubleFunction<M> reducer) {
        var agg = likeKind(1, columnCount());
        return agg.map((_, col, _) -> reducer.applyAsDouble(columnVector(col)));
    }

    /**
     * Returns a new matrix of the same kind as this one, with the specified shape.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @return a new matrix of the same kind as this one, with the specified shape
     */
    M likeKind(int rows, int columns);

    /**
     * Returns the number of columns in the matrix.
     *
     * @return the number of columns
     */
    int columnCount();

    /**
     * Returns the specified column as a column vector.
     *
     * @param column the index of the column to extract
     * @return the column vector
     */
    M columnVector(int column);

    /**
     * Returns the sum of all elements in the matrix.
     *
     * @return the sum of all elements
     */
    default double sum() {
        return values().sum();
    }

    /**
     * Returns the number of rows in the matrix.
     *
     * @return the number of rows
     */
    int rowCount();

    /**
     * Returns the mean value of the specified column.
     *
     * @param column the index of the column
     * @return the mean value of the column
     */
    default double columnMean(int column) {
        return columnValues(column).average().orElseThrow(NoSuchElementException::new);
    }

    /**
     * Returns a row vector containing the standard deviation of each column.
     *
     * @return a new row vector with the standard deviation of each column
     */
    default M columnStd() {
        return reduceColumns(col -> Math.sqrt(col.variance()));
    }

    /**
     * Returns the variance of all elements in the matrix.
     *
     * @return the variance of all elements
     */
    default double variance() {
        return variance(this::values);
    }

    /**
     * Returns the standard deviation of the specified column.
     *
     * @param column the index of the column
     * @return the standard deviation of the column
     */
    default double columnStd(int column) {
        return Math.sqrt(columnVariance(column));
    }

    /**
     * Returns the variance of the specified column.
     *
     * @param column the index of the column
     * @return the variance of the column
     */
    default double columnVariance(int column) {
        return variance(() -> columnValues(column));
    }

    private double variance(Supplier<DoubleStream> stream) {
        var mean = stream.get().average().orElse(0.0);
        return stream.get().map(v -> Math.pow(v - mean, 2)).average().orElse(0.0);
    }

    /**
     * Returns a stream of the values in the specified column.
     *
     * @param column the index of the column
     * @return a stream of the values in the column
     */
    default DoubleStream columnValues(int column) {
        return columnVector(column).values();
    }

    /**
     * Returns a stream of all values in the matrix.
     *
     * @return a stream of all values
     */
    default DoubleStream values() {
        return IntStream.range(0, size()).mapToDouble(index -> valueAt(index / columnCount(), index % columnCount()));
    }

    /**
     * Returns the sum of values in the specified column.
     *
     * @param column the index of the column
     * @return the sum of the column
     */
    default double columnSum(int column) {
        return columnVector(column).sum();
    }

    /**
     * Returns a new row vector containing the variance of each column.
     *
     * @return a new row vector with the variance of each column
     */
    default M columnVariance() {
        return reduceColumns(col -> variance(col::values));
    }

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

    /**
     * Returns a new matrix with each element added to the specified value.
     *
     * @param value the value to add to each element
     * @return a new matrix with each element added to the specified value
     */
    default M elementAdd(double value) {
        return map((_, _, v) -> v + value);
    }

    /**
     * Returns a new matrix with each element divided by the specified value.
     *
     * @param value the value to divide each element by
     * @return a new matrix with each element divided by the specified value
     */
    default M elementDivide(double value) {
        return map((_, _, v) -> v / value);
    }

    /**
     * Returns a new matrix with each element divided by the corresponding element in another matrix.
     *
     * @param other the matrix to divide by
     * @return a new matrix with each element divided by the corresponding element in another matrix
     */
    default M elementDivide(M other) {
        return map((row, col, value) -> value / other.valueAt(row, col));
    }

    /**
     * Multiplies the matrix by another matrix element-wise (hadamard product).
     *
     * @param other the matrix to multiply by
     * @return a new matrix that is the result of the element-wise multiplication (hadamard product)
     */
    default M elementMultiply(M other) {
        return map((row, col, value) -> value * other.valueAt(row, col));
    }

    /**
     * Returns a new matrix with each element raised to the power of 0.5 (square root).
     *
     * @return a new matrix with each element raised to the power of 0.5
     */
    default M elementSqrt() {
        return map((_, _, v) -> Math.sqrt(v));
    }

    /**
     * Returns a new matrix with each element squared.
     *
     * @return a new matrix with each element squared
     */
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

    /**
     * Returns a new matrix with each element subtracted by the specified value.
     *
     * @param value the value to subtract from each element
     * @return a new matrix with each element subtracted by the specified value
     */
    default M elementSubtract(double value) {
        return map((_, _, v) -> v - value);
    }

    /**
     * Returns a stream of the elements in the matrix.
     *
     * @return a stream of the elements in the matrix
     */
    default Stream<MatrixElement> elements() {
        return IntStream.range(0, size()).mapToObj(index -> {
            var row = index / columnCount();
            var column = index % columnCount();
            return new MatrixElement(row, column, valueAt(row, column));
        });
    }

    /**
     * Returns a new matrix with the same shape filled with the specified value.
     *
     * @param value the value to fill the matrix with
     * @return a new matrix with the same shape filled with the specified value
     */
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

    default boolean isEmpty() {
        return rowCount() == 0 || columnCount() == 0;
    }

    /**
     * Checks if the matrix is identical to another matrix within a specified tolerance.
     *
     * @param other     the matrix to compare with
     * @param tolerance the tolerance for comparison
     * @return true if the matrices are identical within the tolerance, false otherwise
     */
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

    /**
     * Returns a new matrix with each element replaced by its natural logarithm.
     *
     * @return a new matrix with each element replaced by its natural logarithm
     */
    default M log() {
        return map((_, _, value) -> Math.log(value));
    }

    /**
     * Returns the maximum value in the matrix.
     *
     * @return the maximum value
     */
    default double max() {
        return deref(values().max());
    }

    private static double deref(OptionalDouble optional) {
        return optional.orElseThrow(NoSuchElementException::new);
    }

    /**
     * Returns the maximum absolute value in the matrix.
     *
     * @return the maximum absolute value
     */
    default double maxAbs() {
        return deref(values().map(Math::abs).max());
    }

    /**
     * Returns the mean of all elements in the matrix.
     *
     * @return the mean of all elements
     */
    default double mean() {
        return deref(values().average());
    }

    /**
     * Returns the minimum value in the matrix.
     *
     * @return the minimum value
     */
    default double min() {
        return deref(values().min());
    }

    default M multiClassifierOutputs(int classCount, int[] labels) {
        if (labels.length != columnCount()) {
            throw new IllegalArgumentException(String.format("Label count %d must match column count %d", labels.length, columnCount()));
        }
        return likeKind(classCount, labels.length)
                .map((row, col, _) -> labels[col] == row ? 1.0 : 0.0);
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
    default M negate() {
        return map((_, _, value) -> -value);
    }

    /**
     * Returns the L1 norm of the matrix.
     *
     * @return the L1 norm
     */
    default double normL1() {
        return sumOfAbs();
    }

    /**
     * Returns the sum of the absolute values of all elements in the matrix.
     *
     * @return the sum of absolute values of all elements
     */
    default double sumOfAbs() {
        return values().map(Math::abs).sum();
    }

    /**
     * Returns the L2 norm of the matrix.
     *
     * @return the L2 norm
     */
    default double normL2() {
        return Math.sqrt(sumOfSquares());
    }

    /**
     * Returns the sum of squares of all elements in the matrix.
     *
     * @return the sum of squares of all elements
     */
    default double sumOfSquares() {
        return values().map(v -> v * v).sum();
    }

    default M normalizeColumn(int col) {
        var max = columnMax(col);
        var min = columnMin(col);

        return map((_, c, value) -> {
            if (c == col) {
                return normalizeValue(value, min, max);
            }
            return value;
        });
    }

    /**
     * Returns the maximum value in the specified column.
     *
     * @param column the index of the column
     * @return the maximum value in the column
     */
    default double columnMax(int column) {
        return columnValues(column).max().orElseThrow(NoSuchElementException::new);
    }

    /**
     * Returns the minimum value in the specified column.
     *
     * @param column the index of the column
     * @return the minimum value in the column
     */
    default double columnMin(int column) {
        return columnValues(column).min().orElseThrow(NoSuchElementException::new);
    }

    private static double normalizeValue(double value, double min, double max) {
        var range = max - min;
        return range <= 0 ? 0.5 : (value - min) / range;
    }

    default M normalizeColumns() {
        var colMax = columnMax();
        var colMin = columnMin();
        return map((_, col, value) -> normalizeValue(value, colMin.valueAt(0, col), colMax.valueAt(0, col)));
    }

    /**
     * Returns a row vector containing the maximum value of each column.
     *
     * @return a new column vector with the maximum value of each column
     */
    default M columnMax() {
        return reduceColumns(Matrix::max);
    }

    /**
     * Returns a row vector containing the minimum value of each column.
     *
     * @return a new row vector with the minimum value of each column
     */
    default M columnMin() {
        return reduceColumns(Matrix::min);
    }

    default M normalizeRow(int row) {
        double min = rowMin(row);
        double max = rowMax(row);

        return map((r, _, value) -> {
            if (r == row) {
                return normalizeValue(value, min, max);
            }
            return value;
        });
    }

    /**
     * Returns the minimum value in the specified row.
     *
     * @param row the index of the row
     * @return the minimum value in the row
     */
    default double rowMin(int row) {
        return deref(rowValues(row).min());
    }

    /**
     * Returns the maximum value in the specified row.
     *
     * @param row the index of the row
     * @return the maximum value in the row
     */
    default double rowMax(int row) {
        return deref(rowValues(row).max());
    }

    default M normalizeRows() {
        var rowMax = rowMax();
        var rowMin = rowMin();
        return map((row, _, value) -> normalizeValue(value, rowMin.valueAt(row, 0), rowMax.valueAt(row, 0)));
    }

    /**
     * Returns a column vector containing the maximum value of each row.
     *
     * @return a new column vector with the maximum value of each row
     */
    default M rowMax() {
        return reduceRows(Matrix::max);
    }

    /**
     * Returns a column vector containing the values provided by the supplied reducer function.
     *
     * @param reducer the reducer function to apply to each row
     * @return a column vector with the values provided by the reducer
     */
    default M reduceRows(ToDoubleFunction<M> reducer) {
        var agg = likeKind(rowCount(), 1);
        return agg.map((row, _, _) -> reducer.applyAsDouble(rowVector(row)));
    }

    /**
     * Returns the specified row as a row vector.
     *
     * @param row the index of the row to extract
     * @return the row vector
     */
    M rowVector(int row);

    /**
     * Returns a column vector containing the minimum value of each row.
     *
     * @return a new column vector with the minimum value of each row
     */
    default M rowMin() {
        return reduceRows(Matrix::min);
    }

    /**
     * Returns the index of the maximum value in the specified row.
     *
     * @param row the index of the row
     * @return the index of the maximum value in the row
     */
    default int rowArgMax(int row) {
        var columnCount = columnCount();
        return IntStream.range(0, columnCount).boxed().max(Comparator.comparingDouble(col -> valueAt(row, col))).orElseThrow(NoSuchElementException::new);
    }

    /**
     * Returns a column vector containing the mean of each row.
     *
     * @return a new column vector with the mean of each row
     */
    default M rowMean() {
        return reduceRows(m -> m.sum() / m.columnCount());
    }

    /**
     * Returns the mean of the specified row.
     *
     * @param row the index of the row
     * @return the mean of the row
     */
    default double rowMean(int row) {
        return deref(rowValues(row).average());
    }

    /**
     * Returns a column vector containing the standard deviation of each row.
     *
     * @return a new column vector with the standard deviation of each row
     */
    default M rowStd() {
        return reduceRows(row -> Math.sqrt(row.variance()));
    }

    /**
     * Returns the standard deviation of the specified row.
     *
     * @param row the index of the row
     * @return the standard deviation of the row
     */
    default double rowStd(int row) {
        return Math.sqrt(rowVariance(row));
    }

    /**
     * Returns the variance of the specified row.
     *
     * @param row the index of the row
     * @return the variance of the row
     */
    default double rowVariance(int row) {
        return variance(() -> rowValues(row));
    }

    /**
     * Returns a stream of the values in the specified row.
     *
     * @param row the index of the row
     * @return a stream of the values in the row
     */
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

    /**
     * Returns the sum of the specified row.
     *
     * @param row the index of the row
     * @return the sum of the row
     */
    default double rowSum(int row) {
        return rowVector(row).sum();
    }

    /**
     * Returns a column vector containing the variance of each row.
     *
     * @return a new column vector with the variance of each row
     */
    default M rowVariance() {
        return reduceRows(row -> variance(row::values));
    }

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
    default M elementMultiply(double scalar) {
        return map((_, _, value) -> value * scalar);
    }

    /**
     * Returns a new matrix with each element replaced by the softmax of the corresponding element.
     *
     * @return a new matrix with softmax applied
     */
    default M softmax() {
        M maxPerColumn = this.columnMax(); // shape: (1, columns)
        M stabilized = this.subtractRowVector(maxPerColumn);
        M exp = stabilized.exp();
        M sumPerColumn = exp.columnSum(); // shape: (1, columns)

        return exp.map((_, col, value) -> value / sumPerColumn.valueAt(0, col));
    }

    /**
     * Returns a new matrix with the specified row vector subtracted from each row of the matrix.
     *
     * @param rowVector the row vector to subtract
     * @return a new matrix with the row vector subtracted from each row
     */
    default M subtractRowVector(M rowVector) {
        return map((_, col, value) -> value - rowVector.valueAt(0, col));
    }

    /**
     * Returns a new matrix with each element replaced by e^x.
     *
     * @return a new matrix with each element replaced by e^x
     */
    default M exp() {
        return map((_, _, value) -> Math.exp(value));
    }

    /**
     * Returns a new column vector containing the sum of each column.
     *
     * @return a new column vector with the sum of each column
     */
    default M columnSum() {
        return reduceColumns(Matrix::sum);
    }

    /**
     * Returns the standard deviation of all elements in the matrix.
     *
     * @return the standard deviation of all elements
     */
    default double std() {
        return Math.sqrt(variance());
    }

    /**
     * Returns a new matrix that is the result of subtracting another matrix from this matrix.
     *
     * @param other the matrix to subtract
     * @return a new matrix that is the result of the subtraction
     */
    default M subtract(M other) {
        return map((row, col, value) -> value - other.valueAt(row, col));
    }

    /**
     * Returns a new matrix that is the result of subtracting a column vector from each column of the matrix.
     *
     * @param columnVector the column vector to subtract
     * @return a new matrix with the column vector subtracted from each column
     */
    default M subtractColumnVector(M columnVector) {
        return map((row, _, value) -> value - columnVector.valueAt(row, 0));
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
