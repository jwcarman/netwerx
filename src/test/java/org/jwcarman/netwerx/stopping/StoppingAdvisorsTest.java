package org.jwcarman.netwerx.stopping;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class StoppingAdvisorsTest {

    @Test
    void testComposite() {
        var advisor = StoppingAdvisors.composite(
                StoppingAdvisors.maxEpoch(100),
                StoppingAdvisors.scoreThreshold(0.01)
        );
        assertThat(advisor.shouldStop(99, 0.00)).isFalse();
        assertThat(advisor.shouldStop(100, 0.00)).isTrue();
        assertThat(advisor.shouldStop(101, 0.00)).isTrue();
        assertThat(advisor.shouldStop(99, 0.01)).isTrue();
        assertThat(advisor.shouldStop(100, 0.01)).isTrue();
        assertThat(advisor.shouldStop(101, 0.01)).isTrue();
    }

}