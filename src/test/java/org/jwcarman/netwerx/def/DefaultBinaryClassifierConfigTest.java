package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.loss.BinaryCrossEntropy;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.Optimizers;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultBinaryClassifierConfigTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void defaultValuesShouldBeSet() {
        var config = new DefaultBinaryClassifierConfig();

        assertThat(config.getLoss()).isInstanceOf(BinaryCrossEntropy.class);
        assertThat(config.getRandom()).isNotNull();
        assertThat(config.getWeightOptimizer()).isNotNull();
        assertThat(config.getBiasOptimizer()).isNotNull();
    }

    @Test
    void fluentSettersShouldWork() {
        var customLoss = Losses.weightedBce(2.0, 1.0);
        var customRandom = new Random(123);
        var weightOpt = Optimizers.momentum(0.01, 0.9);
        var biasOpt = Optimizers.adam();

        var config = new DefaultBinaryClassifierConfig()
                .loss(customLoss)
                .random(customRandom)
                .weightOptimizer(weightOpt)
                .biasOptimizer(biasOpt);

        assertThat(config.getLoss()).isSameAs(customLoss);
        assertThat(config.getRandom()).isSameAs(customRandom);
        assertThat(config.getWeightOptimizer()).isSameAs(weightOpt);
        assertThat(config.getBiasOptimizer()).isSameAs(biasOpt);
    }

    @Test
    void optimizerShouldSetBothWeightAndBiasOptimizers() {
        var sharedOptimizer = Optimizers.rmsProp();

        var config = new DefaultBinaryClassifierConfig()
                .optimizer(sharedOptimizer);

        assertThat(config.getWeightOptimizer()).isSameAs(sharedOptimizer);
        assertThat(config.getBiasOptimizer()).isSameAs(sharedOptimizer);
    }

}
