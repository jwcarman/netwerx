package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class L2NormalizationFunctionTest {

    @Test
    void testNormalize() {
        double[] values = {3.0, 4.0};
        var fn = NormalizationFunctions.l2(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isCloseTo(0.6, withinTolerance());
    }

    @Test
    void testNormalizeWhenNormIsZero() {
        double[] values = {0.0, 0.0};
        var fn = NormalizationFunctions.l2(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isEqualTo(0.0);
    }
}