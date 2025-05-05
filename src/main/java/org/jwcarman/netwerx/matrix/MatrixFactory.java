package org.jwcarman.netwerx.matrix;

import java.util.List;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;

public interface MatrixFactory<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Creates an empty (zero-sized) matrix.
     *
     * @return a new empty matrix instance
     */
    M empty();

    /**
     * Creates a new matrix from the specified data.
     *
     * @param data the data for the matrix
     * @return a new matrix filled with the specified data
     */
    M from(double[][] data);

    /**
     * Creates a new matrix with the specified data.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @param values  the values to fill the matrix with
     * @return a new matrix with the specified dimensions and filled with the values
     */
    default M from(int rows, int columns, double... values) {
        if (values.length != rows * columns) {
            throw new IllegalArgumentException(String.format("Invalid number of values (%d) for the specified dimensions (%d x %d), expecting %d values.", values.length, rows, columns, rows * columns));
        }
        return filled(rows, columns, (row, col) -> values[row * columns + col]);
    }

    /**
     * Creates a new matrix with the specified data.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @param values  a supplier for the values to fill the matrix with
     * @return a new matrix with the specified dimensions and filled with values from the supplier
     */
    M filled(int rows, int columns, MatrixValueProvider values);

    /**
     * Creates a new matrix filled with Gaussian-distributed random values.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @param mean    the mean of the Gaussian distribution
     * @param stddev  the standard deviation of the Gaussian distribution
     * @param random  a Random instance to generate random values
     * @return a new matrix with the specified dimensions and filled with Gaussian-distributed random values
     */
    default M gaussian(int rows, int columns, double mean, double stddev, Random random) {
        return filled(rows, columns, () -> random.nextGaussian() * stddev + mean);
    }

    /**
     * Creates a new matrix with the specified data.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @param values  a supplier for the values to fill the matrix with
     * @return a new matrix with the specified dimensions and filled with values from the supplier
     */
    default M filled(int rows, int columns, DoubleSupplier values) {
        return filled(rows, columns, (_, _) -> values.getAsDouble());
    }

    /**
     * Creates a new matrix filled with ones.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @return a new matrix with the specified dimensions and filled with ones
     */
    default M ones(int rows, int columns) {
        return filled(rows, columns, 1.0);
    }

    /**
     * Creates a new matrix filled with the specified constant value.
     *
     * @param rows          the number of rows
     * @param columns       the number of columns
     * @param constantValue the constant value to fill the matrix with
     * @return a new matrix with the specified dimensions and filled with the constant value
     */
    M filled(int rows, int columns, double constantValue);

    /**
     * Creates a new matrix filled with random values.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @param random  a Random instance to generate random values
     * @return a new matrix with the specified dimensions and filled with random values
     */
    default M random(int rows, int columns, Random random) {
        return filled(rows, columns, (_, _) -> random.nextDouble());
    }

    /**
     * Creates a new matrix filled with random values within a specified range.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @param min     the minimum value (inclusive)
     * @param max     the maximum value (exclusive)
     * @param random  a Random instance to generate random values
     * @return a new matrix with the specified dimensions and filled with random values within the range
     */
    default M random(int rows, int columns, double min, double max, Random random) {
        final var range = max - min;
        return filled(rows, columns, () -> min + (range * random.nextDouble()));
    }


    /**
     * Creates a new matrix with the specified column values and row mappers.
     *
     * @param columnValues the values for each column
     * @param rowMappers   a list of functions that map each column value to a double for each row
     * @param <T>          the type of the column values
     * @return a new matrix with the specified column values and row mappers
     */
    default <T> M columnOriented(List<T> columnValues, List<ToDoubleFunction<T>> rowMappers) {
        return filled(rowMappers.size(), columnValues.size(), (row, col) -> rowMappers.get(row).applyAsDouble(columnValues.get(col)));
    }

    /**
     * Creates a new matrix with the specified row values and column mappers.
     *
     * @param rowValues     the values for each row
     * @param columnMappers a list of functions that map each row value to a double for each column
     * @param <T>           the type of the row values
     * @return a new matrix with the specified row values and column mappers
     */
    default <T> M rowOriented(List<T> rowValues, List<ToDoubleFunction<T>> columnMappers) {
        return filled(rowValues.size(), columnMappers.size(), (row, col) -> columnMappers.get(col).applyAsDouble(rowValues.get(row)));
    }

    /**
     * Creates a new matrix with the specified number of rows and columns.
     *
     * @param rows    the number of rows
     * @param columns the number of columns
     * @return a new matrix with the specified dimensions (filled with zeros)
     */
    M zeros(int rows, int columns);

}
