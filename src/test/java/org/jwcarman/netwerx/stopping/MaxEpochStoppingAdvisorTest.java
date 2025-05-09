package org.jwcarman.netwerx.stopping;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class MaxEpochStoppingAdvisorTest {

    @Test
    void testDefaultConstructor() {
        var advisor = StoppingAdvisors.maxEpoch();
        assertThat(advisor.shouldStop(99, 100.00)).isFalse();
        assertThat(advisor.shouldStop(100, 100.00)).isTrue();
        assertThat(advisor.shouldStop(101, 100.00)).isTrue();
    }

    @Test
    void testConstructorWithMaxEpoch() {
        var advisor = StoppingAdvisors.maxEpoch(50);
        assertThat(advisor.shouldStop(49, 100.00)).isFalse();
        assertThat(advisor.shouldStop(50, 100.00)).isTrue();
        assertThat(advisor.shouldStop(51, 100.00)).isTrue();
    }

}