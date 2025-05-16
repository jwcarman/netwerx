package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class ZScoreNormalizationFunctionTest {

    @Test
    void testNormalize() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var fn = NormalizationFunctions.zScore(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isCloseTo(0.0, withinTolerance());
    }

}