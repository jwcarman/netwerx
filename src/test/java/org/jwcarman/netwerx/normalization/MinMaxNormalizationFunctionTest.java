package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class MinMaxNormalizationFunctionTest {
    @ParameterizedTest
    @CsvSource({
            "3.0, 0.0",
            "5.0, 1.0",
            "2.0, 0.0",
            "-4.0, 0.0"
    })
    void testNormalize(double value, double expected) {
        double[] values = {3.0, 4.0};
        var fn = NormalizationFunctions.minMax(Arrays.stream(values));
        assertThat(fn.normalize(value)).isCloseTo(expected, withinTolerance());
    }

    @Test
    void testNormalizeValueWithNoRange() {
        double[] values = {4.0};
        var fn = NormalizationFunctions.minMax(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isCloseTo(0.5, withinTolerance());
    }

}