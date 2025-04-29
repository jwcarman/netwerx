package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class SoftmaxTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void apply_shouldBeNumericallyStableForLargeValues() {
        var softmax = Activations.softmax();
        var input = new SimpleMatrix(3, 1, true, 1.0, 2.0, 100.0);
        var output = softmax.apply(input);

        // Find index of max value
        int maxIdx = 0;
        double maxVal = output.get(0, 0);
        for (int i = 1; i < output.getNumRows(); i++) {
            if (output.get(i, 0) > maxVal) {
                maxVal = output.get(i, 0);
                maxIdx = i;
            }
        }

        // Confirm maxIdx corresponds to input value 100.0 (row 2)
        assertThat(maxIdx).isEqualTo(2);

        // Confirm softmax probabilities still sum to 1
        double sum = output.elementSum();
        assertThat(sum).isCloseTo(1.0, within(1e-6));
    }

    @Test
    void apply_shouldOutputProbabilitiesSummingToOnePerColumn() {
        var softmax = Activations.softmax();
        var input = new SimpleMatrix(3, 2, true, 1.0, 5.0, 2.0, 4.0, 3.0, 6.0);
        var output = softmax.apply(input);

        for (int col = 0; col < output.getNumCols(); col++) {
            double columnSum = 0.0;
            for (int row = 0; row < output.getNumRows(); row++) {
                columnSum += output.get(row, col);
            }
            assertThat(columnSum).isCloseTo(1.0, within(1e-6));
        }
    }

    @Test
    void apply_shouldOutputSameShape() {
        var softmax = Activations.softmax();
        var input = new SimpleMatrix(3, 2, true, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0); // 3 rows, 2 columns
        var output = softmax.apply(input);

        assertThat(output.getNumRows()).isEqualTo(3);
        assertThat(output.getNumCols()).isEqualTo(2);
    }

}
