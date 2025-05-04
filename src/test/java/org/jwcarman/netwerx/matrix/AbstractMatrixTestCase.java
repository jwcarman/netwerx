package org.jwcarman.netwerx.matrix;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Tolerances;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertEquals;

public abstract class AbstractMatrixTestCase<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    protected abstract MatrixFactory<M> factory();

    @Test
    void testIsEmpty() {
        assertThat(factory().zeros(0, 0).isEmpty()).isTrue();
        assertThat(factory().zeros(1, 0).isEmpty()).isTrue();
        assertThat(factory().zeros(0, 1).isEmpty()).isTrue();
    }

    @Test
    void testNormalizeColumn() {
        M matrix = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        M result = matrix.normalizeColumn(0);
        assertThat(result.values().boxed().toList()).containsExactly(0.0, 2.0, 3.0, 1.0, 5.0, 6.0);
    }

    @Test
    void testNormalizeColumns() {
        M matrix = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        M result = matrix.normalizeColumns();
        assertThat(result.values().boxed().toList()).containsExactly(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    }

    @Test
    void testRowSlice() {
        M matrix = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        assertThatThrownBy(() -> matrix.rowSlice(-4, 2))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> matrix.rowSlice(0, 5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> matrix.rowSlice(2, 1))
                .isInstanceOf(IllegalArgumentException.class);

        var slice = matrix.rowSlice(0, 1);
        assertThat(slice.rowCount()).isEqualTo(1);
        assertThat(slice.columnCount()).isEqualTo(matrix.columnCount());
    }

    @Test
    void testColumnSlice() {
        M matrix = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        assertThatThrownBy(() -> matrix.columnSlice(-4, 2))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> matrix.columnSlice(0, 5))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> matrix.columnSlice(2, 1))
                .isInstanceOf(IllegalArgumentException.class);

        var slice = matrix.columnSlice(0, 1);
        assertThat(slice.rowCount()).isEqualTo(matrix.rowCount());
        assertThat(slice.columnCount()).isEqualTo(1);
    }

    @Test
    void testNormalizeColumnWithZeroRange() {
        M matrix = factory().from(2, 3, 1, 2, 3, 1, 5, 6);
        M result = matrix.normalizeColumn(0);
        assertThat(result.values().boxed().toList()).containsExactly(0.5, 2.0, 3.0, 0.5, 5.0, 6.0);
    }

    @Test
    void testNormalizeRow() {
        M matrix = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        M result = matrix.normalizeRow(0);
        assertThat(result.values().boxed().toList()).containsExactly(0.0, 0.5, 1.0, 4.0, 5.0, 6.0);
    }

    @Test
    void testNormalizeRows() {
        M matrix = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        M result = matrix.normalizeRows();
        assertThat(result.values().boxed().toList()).containsExactly(0.0, 0.5, 1.0, 0.0, 0.5, 1.0);
    }

    @Test
    void testNormalizeRowWithZeroRange() {
        M matrix = factory().from(2, 3, 1, 1, 1, 4, 5, 6);
        M result = matrix.normalizeRow(0);
        assertThat(result.values().boxed().toList()).containsExactly(0.5, 0.5, 0.5, 4.0, 5.0, 6.0);
    }

    @Test
    void testMultiClassifierOutput() {
        var m = factory().zeros(3, 3)
                .multiClassifierOutputs(3, new int[]{0, 1, 2});
        assertThat(m.values().boxed()).containsExactly(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0);
    }

    @Test
    void testMultiClassifierOutputWithInvalidDims() {
        var matrix = factory().zeros(3, 1);
        var labels = new int[]{0, 1, 2};

        assertThatThrownBy(() -> matrix.multiClassifierOutputs(3, labels))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void testBinaryClassifierOutputs() {
        var m = factory().zeros(3, 3)
                .binaryClassifierOutputs(new boolean[]{true, false, true});
        assertThat(m.values().boxed()).containsExactly(1.0, 0.0, 1.0);
    }

    @Test
    void testBinaryClassifierOutputsWithInvalidDims() {
        var matrix = factory().zeros(3, 1);
        var labels = new boolean[]{true, false, true};

        assertThatThrownBy(() -> matrix.binaryClassifierOutputs(labels))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void testAdd() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(2, 2, 2.0);
        M expected = factory().filled(2, 2, 3.0);

        M result = a.add(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testAddColumnVector() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(2, 1, 1.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.addColumnVector(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testAddRowVector() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(1, 2, 1.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.addRowVector(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnMax() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 1.0);

        M result = a.columnMax();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnMaxForSpecificColumn() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.columnMax(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testColumnMean() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 1.0);

        M result = a.columnMean();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnMeanForSpecificColumn() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.columnMean(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testColumnMin() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 1.0);

        M result = a.columnMin();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnMinForSpecificColumn() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.columnMin(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testColumnStd() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 0.0);

        M result = a.columnStd();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnStdForSpecificColumn() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 0.0;

        double result = a.columnStd(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testColumnSum() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 2.0);

        M result = a.columnSum();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnSumForSpecificColumn() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 2.0;

        double result = a.columnSum(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testColumnVariance() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 0.0);

        M result = a.columnVariance();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testColumnVarianceForSpecificColumn() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 0.0;

        double result = a.columnVariance(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testColumnVector() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 1.0);

        M result = a.columnVector(0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testCopy() {
        M a = factory().filled(2, 2, 1.0);
        M expected = a.copy();

        M result = a.copy();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testElementAbs() {
        M a = factory().filled(2, 2, -1.0);
        M expected = factory().filled(2, 2, 1.0);

        M result = a.elementAbs();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testElementDivideByScalar() {
        M a = factory().filled(2, 2, 4.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.elementDivide(2.0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testElementPower() {
        M a = factory().filled(2, 2, 2.0);
        M expected = factory().filled(2, 2, 8.0);

        M result = a.elementPower(3.0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testElementSqrt() {
        M a = factory().filled(2, 2, 4.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.elementSqrt();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testElementSquare() {
        M a = factory().filled(2, 2, 2.0);
        M expected = factory().filled(2, 2, 4.0);

        M result = a.elementSquare();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testElementSubtract() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 2, -1.0);

        M result = a.elementSubtract(2.0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testFillWithScalar() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 2, 1.0);

        M result = a.fill(1.0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testFlattenToColumn() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(4, 1, 1.0);

        M result = a.flattenToColumn();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testFlattenToRow() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 4, 1.0);

        M result = a.flattenToRow();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testIsNotSquare() {
        M a = factory().filled(2, 3, 1.0);
        boolean expected = false;

        boolean result = a.isSquare();

        assertEquals(expected, result);
    }

    @Test
    void testIsSquare() {
        M a = factory().filled(2, 2, 1.0);
        boolean expected = true;

        boolean result = a.isSquare();

        assertEquals(expected, result);
    }

    @Test
    void testLikeKind() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 3, 0.0);

        M result = a.likeKind(2, 3);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    public void testMatrixElementWiseDivision() {
        M a = factory().filled(2, 2, 4.0);
        M b = factory().filled(2, 2, 2.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.elementDivide(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    public void testMatrixElementWiseMultiplication() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(2, 2, 2.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.elementMultiply(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    public void testMatrixMultiplication() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(2, 2, 2.0);
        M expected = factory().filled(2, 2, 4.0);

        M result = a.multiply(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    public void testMatrixScale() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 2, 2.0);

        M result = a.scale(2.0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    public void testMatrixSum() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 4.0;

        double result = a.sum();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    public void testMatrixTranspose() {
        M a = factory().filled(2, 3, 1.0);
        M expected = factory().filled(3, 2, 1.0);

        M result = a.transpose();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testMaxAbs() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.maxAbs();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testMean() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.mean();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testNormL1() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 4.0;

        double result = a.normL1();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testNormL2() {
        M a = factory().filled(2, 2, 1.0);
        double expected = Math.sqrt(4);

        double result = a.normL2();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testReshape() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(4, 1, 1.0);

        M result = a.reshape(4, 1);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowArgMaxForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        int expected = 0;

        int result = a.rowArgMax(0);

        assertEquals(expected, result);
    }

    @Test
    void testRowMax() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 1.0);

        M result = a.rowMax();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowMaxForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.rowMax(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testRowMean() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 1.0);

        M result = a.rowMean();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowMeanForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.rowMean(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testRowMin() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 1.0);

        M result = a.rowMin();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowMinForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 1.0;

        double result = a.rowMin(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testRowStd() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 0.0);

        M result = a.rowStd();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowStdForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 0.0;

        double result = a.rowStd(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testRowSum() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 2.0);

        M result = a.rowSum();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowSumForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 2.0;

        double result = a.rowSum(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testRowVariance() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(2, 1, 0.0);

        M result = a.rowVariance();

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testRowVarianceForSpecificRow() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 0.0;

        double result = a.rowVariance(0);

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testRowVector() {
        M a = factory().filled(2, 2, 1.0);
        M expected = factory().filled(1, 2, 1.0);

        M result = a.rowVector(0);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testStd() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 0.0;

        double result = a.std();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

    @Test
    void testSubtractColumnVector() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(2, 1, 1.0);
        M expected = factory().filled(2, 2, 0.0);

        M result = a.subtractColumnVector(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testSubtractRowVector() {
        M a = factory().filled(2, 2, 1.0);
        M b = factory().filled(1, 2, 1.0);
        M expected = factory().filled(2, 2, 0.0);

        M result = a.subtractRowVector(b);

        assertThat(expected.isIdentical(result, Tolerances.DEFAULT_TOLERANCE)).isTrue();
    }

    @Test
    void testValues() {
        var m = factory().from(2, 3, 1, 2, 3, 4, 5, 6);
        var result = m.values().boxed().toList();
        assertThat(result).containsExactly(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    @Test
    void testVariance() {
        M a = factory().filled(2, 2, 1.0);
        double expected = 0.0;

        double result = a.variance();

        assertEquals(expected, result, Tolerances.DEFAULT_TOLERANCE);
    }

}
