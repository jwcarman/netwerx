package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.loss.MeanSquaredError;
import org.jwcarman.netwerx.optimization.AdamOptimizer;
import org.jwcarman.netwerx.optimization.RmsPropOptimizer;
import org.jwcarman.netwerx.optimization.SgdOptimizer;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultRegressionModelConfigTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void defaultValuesShouldBeSetCorrectly() {
        var config = new DefaultRegressionModelConfig();

        assertThat(config.getLoss()).isInstanceOf(MeanSquaredError.class);
        assertThat(config.getWeightOptimizer()).isInstanceOf(SgdOptimizer.class);
        assertThat(config.getBiasOptimizer()).isInstanceOf(SgdOptimizer.class);
        assertThat(config.getRandom()).isNotNull();
    }

    @Test
    void fluentSettersShouldUpdateConfiguration() {
        var customLoss = Losses.bce();
        var random = new Random(123);
        var weightOpt = new AdamOptimizer();
        var biasOpt = new RmsPropOptimizer();

        var config = new DefaultRegressionModelConfig()
                .loss(customLoss)
                .random(random)
                .weightOptimizer(weightOpt)
                .biasOptimizer(biasOpt);

        assertThat(config.getLoss()).isSameAs(customLoss);
        assertThat(config.getWeightOptimizer()).isSameAs(weightOpt);
        assertThat(config.getBiasOptimizer()).isSameAs(biasOpt);
        assertThat(config.getRandom()).isSameAs(random);
    }

    @Test
    void optimizerShouldSetBothWeightAndBiasOptimizers() {
        var shared = new RmsPropOptimizer();

        var config = new DefaultRegressionModelConfig()
                .optimizer(shared);

        assertThat(config.getWeightOptimizer()).isSameAs(shared);
        assertThat(config.getBiasOptimizer()).isSameAs(shared);
    }

}
