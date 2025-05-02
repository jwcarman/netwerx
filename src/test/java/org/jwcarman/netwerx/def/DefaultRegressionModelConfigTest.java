package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.loss.MeanSquaredError;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultRegressionModelConfigTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void defaultValuesShouldBeSetCorrectly() {
        var config = new DefaultRegressionModelConfig();

        assertThat(config.getLoss()).isInstanceOf(MeanSquaredError.class);
    }

    @Test
    void fluentSettersShouldUpdateConfiguration() {
        var customLoss = Losses.bce();

        var config = new DefaultRegressionModelConfig()
                .loss(customLoss);

        assertThat(config.getLoss()).isSameAs(customLoss);
    }
}
