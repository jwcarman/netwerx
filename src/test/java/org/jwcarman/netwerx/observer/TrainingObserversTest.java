package org.jwcarman.netwerx.observer;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatNoException;

class TrainingObserversTest {

    @Test
    void testNoop() {
        var observer = TrainingObservers.noop();
        assertThatNoException()
                .isThrownBy(() -> observer.onEpoch(null));
    }

}