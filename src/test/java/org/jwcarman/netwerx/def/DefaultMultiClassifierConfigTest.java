package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.loss.CategoricalCrossEntropy;
import org.jwcarman.netwerx.loss.Losses;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultMultiClassifierConfigTest {
    @Test
    void defaultValuesShouldBeSetCorrectly() {
        var config = new DefaultMultiClassifierConfig();

        assertThat(config.getOutputClasses()).isEqualTo(3);
        assertThat(config.getLoss()).isInstanceOf(CategoricalCrossEntropy.class);
    }

    @Test
    void fluentSettersShouldUpdateConfiguration() {
        var loss = Losses.mae();

        var config = new DefaultMultiClassifierConfig()
                .outputClasses(5)
                .loss(loss);

        assertThat(config.getOutputClasses()).isEqualTo(5);
        assertThat(config.getLoss()).isSameAs(loss);
    }
}