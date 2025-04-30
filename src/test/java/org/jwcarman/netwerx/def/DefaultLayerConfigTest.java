package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.activation.ReLU;
import org.jwcarman.netwerx.optimization.Optimizers;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultLayerConfigTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void defaultValuesShouldBeSet() {
        var config = new DefaultLayerConfig(6);

        assertThat(config.getInputSize()).isEqualTo(6);
        assertThat(config.getUnits()).isEqualTo(10);
        assertThat(config.getActivation()).isInstanceOf(ReLU.class);
        assertThat(config.getWeightOptimizer()).isNotNull();
        assertThat(config.getBiasOptimizer()).isNotNull();
        assertThat(config.getRandom()).isNotNull();
    }

    @Test
    void fluentSettersShouldUpdateConfiguration() {
        var random = new Random(42);
        var activation = Activations.sigmoid();
        var weightOpt = Optimizers.adam();
        var biasOpt = Optimizers.rmsProp();

        var config = new DefaultLayerConfig(5)
                .units(32)
                .activation(activation)
                .random(random)
                .weightOptimizer(weightOpt)
                .biasOptimizer(biasOpt);

        assertThat(config.getInputSize()).isEqualTo(5);
        assertThat(config.getUnits()).isEqualTo(32);
        assertThat(config.getActivation()).isSameAs(activation);
        assertThat(config.getWeightOptimizer()).isSameAs(weightOpt);
        assertThat(config.getBiasOptimizer()).isSameAs(biasOpt);
        assertThat(config.getRandom()).isSameAs(random);
    }

    @Test
    void optimizerShouldSetBothWeightAndBiasOptimizers() {
        var shared = Optimizers.momentum(0.05, 0.9);

        var config = new DefaultLayerConfig(4)
                .optimizer(shared);

        assertThat(config.getWeightOptimizer()).isSameAs(shared);
        assertThat(config.getBiasOptimizer()).isSameAs(shared);
    }

}
