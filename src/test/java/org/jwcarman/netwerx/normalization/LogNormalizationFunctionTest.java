package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class LogNormalizationFunctionTest {

    @Test
    void testNormalize() {
        double[] values = {1.0, 10.0, 100.0};

        var fn = NormalizationFunctions.log(Arrays.stream(values));

        double normalizedValue = fn.normalize(10.0);
        assertThat(normalizedValue).isCloseTo(Math.log1p(10.0 - 1.0), withinTolerance());
    }

    @Test
    void testNormalizeNegativeValue() {
        double[] values = {1.0, 10.0, 100.0};

        var fn = NormalizationFunctions.log(Arrays.stream(values));

        double normalizedValue = fn.normalize(-5.0);
        assertThat(normalizedValue).isEqualTo(0.0);
    }

}