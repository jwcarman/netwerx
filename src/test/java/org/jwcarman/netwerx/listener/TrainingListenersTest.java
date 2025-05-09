package org.jwcarman.netwerx.listener;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatNoException;

class TrainingListenersTest {

    @Test
    void testNoop() {
        var observer = TrainingListeners.noop();
        assertThatNoException()
                .isThrownBy(() -> observer.onEpoch(null));
    }

}