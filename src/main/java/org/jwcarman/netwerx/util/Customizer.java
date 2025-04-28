package org.jwcarman.netwerx.util;

public interface Customizer<C> {

    static <C> Customizer<C> noop() {
        return config -> {
            // No operation
        };
    }

    void customize(C config);
}
