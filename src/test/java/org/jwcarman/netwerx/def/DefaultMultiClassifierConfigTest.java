package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.loss.CategoricalCrossEntropy;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.AdamOptimizer;
import org.jwcarman.netwerx.optimization.RmsPropOptimizer;
import org.jwcarman.netwerx.optimization.SgdOptimizer;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultMultiClassifierConfigTest {
    @Test
    void defaultValuesShouldBeSetCorrectly() {
        var config = new DefaultMultiClassifierConfig();

        assertThat(config.getOutputClasses()).isEqualTo(3);
        assertThat(config.getLoss()).isInstanceOf(CategoricalCrossEntropy.class);
        assertThat(config.getWeightOptimizer()).isInstanceOf(SgdOptimizer.class);
        assertThat(config.getBiasOptimizer()).isInstanceOf(SgdOptimizer.class);
        assertThat(config.getRandom()).isNotNull();
    }

    @Test
    void fluentSettersShouldUpdateConfiguration() {
        var loss = Losses.mae();
        var random = new Random(123);
        var weightOpt = new AdamOptimizer();
        var biasOpt = new RmsPropOptimizer();

        var config = new DefaultMultiClassifierConfig()
                .outputClasses(5)
                .loss(loss)
                .random(random)
                .weightOptimizer(weightOpt)
                .biasOptimizer(biasOpt);

        assertThat(config.getOutputClasses()).isEqualTo(5);
        assertThat(config.getLoss()).isSameAs(loss);
        assertThat(config.getWeightOptimizer()).isSameAs(weightOpt);
        assertThat(config.getBiasOptimizer()).isSameAs(biasOpt);
        assertThat(config.getRandom()).isSameAs(random);
    }

    @Test
    void optimizerShouldSetBothWeightAndBiasOptimizers() {
        var shared = new RmsPropOptimizer();

        var config = new DefaultMultiClassifierConfig()
                .optimizer(shared);

        assertThat(config.getWeightOptimizer()).isSameAs(shared);
        assertThat(config.getBiasOptimizer()).isSameAs(shared);
    }
}