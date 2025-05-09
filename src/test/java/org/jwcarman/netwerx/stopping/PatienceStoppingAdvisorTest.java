package org.jwcarman.netwerx.stopping;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PatienceStoppingAdvisorTest {

    @Test
    void testDefaultConstructor() {
        var advisor = StoppingAdvisors.patience();
        assertThat(advisor.shouldStop(1, 100.00000)).isFalse();
        assertThat(advisor.shouldStop(2, 100.00001)).isFalse();
        assertThat(advisor.shouldStop(3, 100.00002)).isFalse();
        assertThat(advisor.shouldStop(4, 100.00003)).isFalse();
        assertThat(advisor.shouldStop(5, 100.00004)).isFalse();
        assertThat(advisor.shouldStop(6, 100.00005)).isFalse();
        assertThat(advisor.shouldStop(7, 100.00006)).isFalse();
        assertThat(advisor.shouldStop(8, 100.00007)).isFalse();
        assertThat(advisor.shouldStop(9, 100.00008)).isFalse();
        assertThat(advisor.shouldStop(10, 100.00009)).isFalse();
        assertThat(advisor.shouldStop(11, 100.00010)).isTrue();
    }

    @Test
    void testConstructorWithPatience() {
        var advisor = StoppingAdvisors.patience(5, PatienceStoppingAdvisor.DEFAULT_MIN_DELTA);
        assertThat(advisor.shouldStop(1, 100.00000)).isFalse();
        assertThat(advisor.shouldStop(2, 100.00001)).isFalse();
        assertThat(advisor.shouldStop(3, 100.00002)).isFalse();
        assertThat(advisor.shouldStop(4, 100.00003)).isFalse();
        assertThat(advisor.shouldStop(5, 100.00004)).isFalse();
        assertThat(advisor.shouldStop(6, 100.00005)).isTrue();
    }

    @Test
    void testConstructorWithPatienceAndMinDelta() {
        var advisor = StoppingAdvisors.patience(5, 0.00001);
        assertThat(advisor.shouldStop(1, 100.000000)).isFalse();
        assertThat(advisor.shouldStop(2, 100.000001)).isFalse();
        assertThat(advisor.shouldStop(3, 100.000002)).isFalse();
        assertThat(advisor.shouldStop(4, 100.000003)).isFalse();
        assertThat(advisor.shouldStop(5, 100.000004)).isFalse();
        assertThat(advisor.shouldStop(6, 100.000005)).isTrue();
    }

}