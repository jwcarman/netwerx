package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class MaxAbsNormalizationFunctionTest {

    @Test
    void testNormalize() {
        double[] values = {3.0, 4.0};
        var fn = NormalizationFunctions.maxAbs(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isCloseTo(0.75, withinTolerance());
    }

    @Test
    void testNormalizeNegativeValue() {
        double[] values = {3.0, 4.0};
        var fn = NormalizationFunctions.maxAbs(Arrays.stream(values));

        double normalizedValue = fn.normalize(-5.0);
        assertThat(normalizedValue).isCloseTo(-1.25, withinTolerance());
    }

    @Test
    void testNormalizeWithMaxAbsZero() {
        double[] values = {0.0, 0.0};
        var fn = NormalizationFunctions.maxAbs(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isEqualTo(0.0);
    }
}