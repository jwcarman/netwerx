package org.jwcarman.netwerx.stopping;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class ScoreThresholdStoppingAdvisorTest {

    @Test
    void testDefaultConstructor() {
        var advisor = StoppingAdvisors.scoreThreshold();
        assertThat(advisor.shouldStop(99, -0.001)).isTrue();
        assertThat(advisor.shouldStop(100, -0.01)).isTrue();
        assertThat(advisor.shouldStop(101, -0.1)).isFalse();
    }

}