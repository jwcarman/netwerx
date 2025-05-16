package org.jwcarman.netwerx.normalization;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class NormalizationFunctionsTest {
    @Test
    void testIdentity() {
        var fn = NormalizationFunctions.identity();
        assertThat(fn.normalize(6.8912)).isEqualTo(6.8912);
    }

}