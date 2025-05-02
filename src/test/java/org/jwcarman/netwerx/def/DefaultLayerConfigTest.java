package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.activation.ReLU;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultLayerConfigTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void defaultValuesShouldBeSet() {
        var config = new DefaultLayerConfig(6);

        assertThat(config.getInputSize()).isEqualTo(6);
        assertThat(config.getUnits()).isEqualTo(10);
        assertThat(config.getActivation()).isInstanceOf(ReLU.class);
    }

    @Test
    void fluentSettersShouldUpdateConfiguration() {
        var activation = Activations.sigmoid();

        var config = new DefaultLayerConfig(5)
                .units(32)
                .activation(activation);

        assertThat(config.getInputSize()).isEqualTo(5);
        assertThat(config.getUnits()).isEqualTo(32);
        assertThat(config.getActivation()).isSameAs(activation);
    }

}
