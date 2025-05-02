package org.jwcarman.netwerx.activation;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.util.Matrices;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class ELUTest {
    @Test
    void apply_shouldApplyELUFunctionToEachElement_defaultAlpha() {
        var elu = Activations.elu(); // default alpha = 1.0
        var input = Matrices.of(new double[][]{
                {1.0, 0.0, -1.0}
        });

        var result = elu.apply(input);

        assertThat(result.valueAt(0, 0)).isEqualTo(1.0);
        assertThat(result.valueAt(0, 1)).isEqualTo(0.0);
        assertThat(result.valueAt(0, 2)).isCloseTo(Math.expm1(-1.0), withinTolerance());
    }

    @Test
    void apply_shouldApplyELUFunctionToEachElement_customAlpha() {
        var alpha = 2.0;
        var elu = Activations.elu(0.0, alpha);
        var input = Matrices.of(new double[][]{
                {2.0, -0.5}
        });

        var result = elu.apply(input);

        assertThat(result.valueAt(0, 0)).isEqualTo(2.0);
        assertThat(result.valueAt(0, 1)).isCloseTo(alpha * (Math.exp(-0.5) - 1), withinTolerance());
    }

    @Test
    void derivative_shouldApplyDerivativeOfELUFunction_defaultAlpha() {
        var elu = Activations.elu(); // alpha = 1.0
        var input = Matrices.of(new double[][]{
                {2.0, 0.0, -1.0}
        });

        var result = elu.derivative(input);

        assertThat(result.valueAt(0, 0)).isEqualTo(1.0);
        assertThat(result.valueAt(0, 1)).isEqualTo(1.0);
        assertThat(result.valueAt(0, 2)).isCloseTo(Math.exp(-1.0), withinTolerance());
    }

    @Test
    void derivative_shouldApplyDerivativeOfELUFunction_customAlpha() {
        var alpha = 1.5;
        var elu = Activations.elu(0.0, alpha);
        var input = Matrices.of(new double[][]{
                {-2.0, 1.0}
        });

        var result = elu.derivative(input);

        assertThat(result.valueAt(0, 0)).isCloseTo(alpha * Math.exp(-2.0), withinTolerance());
        assertThat(result.valueAt(0, 1)).isEqualTo(1.0);
    }
}