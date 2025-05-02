package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.loss.BinaryCrossEntropy;
import org.jwcarman.netwerx.loss.Losses;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultBinaryClassifierConfigTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void defaultValuesShouldBeSet() {
        var config = new DefaultBinaryClassifierConfig();

        assertThat(config.getLoss()).isInstanceOf(BinaryCrossEntropy.class);
    }

    @Test
    void fluentSettersShouldWork() {
        var customLoss = Losses.weightedBce(2.0, 1.0);
        var config = new DefaultBinaryClassifierConfig()
                .loss(customLoss);

        assertThat(config.getLoss()).isSameAs(customLoss);
    }
}
