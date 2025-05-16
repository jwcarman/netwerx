package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class RobustScalingNormalizationFunctionTest {

    public static final double[] EMPTY_ARRAY = {};

    @Test
    void testNormalize() {
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        var fn = NormalizationFunctions.robustScaling(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isCloseTo(0.0, withinTolerance());
    }

    @Test
    void testNormalizeWithEvenNumberOfValues() {
        double[] values = {1.0, 2.0, 3.0, 4.0};
        var fn = NormalizationFunctions.robustScaling(Arrays.stream(values));

        double normalizedValue = fn.normalize(3.0);
        assertThat(normalizedValue).isCloseTo(1.0 / 3.0, withinTolerance());
    }

    @Test
    void testNormalizeWithNoValues() {
        var values = Arrays.stream(EMPTY_ARRAY);
        assertThatThrownBy(() -> NormalizationFunctions.robustScaling(values))
                .isInstanceOf(IllegalArgumentException.class);
    }

}