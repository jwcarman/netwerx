package org.jwcarman.netwerx.network;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.stopping.StoppingAdvisors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DefaultNeuralNetworkTrainerTest {

    @Test
    void emptyValidationSetShouldYieldNaN() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 2)
                .listener(outcome -> assertThat(outcome.validationLoss()).isNaN())
                .stoppingAdvisor(StoppingAdvisors.maxEpoch(1))
                .denseLayer()
                .denseLayer(layer -> layer.units(1))
                .build();

        var inputs = factory.filled(2, 2, 1.0);
        var targets = factory.filled(1, 2, 10.0);
        var dataset = new Dataset<>(inputs, targets);
        trainer.train(dataset);
    }

    @Test
    void trainingShouldValidatesInputSize() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .denseLayer()
                .denseLayer(layer -> layer.units(1))
                .build();

        var inputs = factory.filled(2, 2, 1.0);
        var targets = factory.filled(1, 2, 10.0);
        var dataset = new Dataset<>(inputs, targets);
        assertThatThrownBy(() -> trainer.train(dataset))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void inferenceShouldValidateInputSize() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 2)
                .denseLayer()
                .denseLayer(layer -> layer.units(1))
                .build();

        var inputs = factory.filled(2, 2, 1.0);
        var targets = factory.filled(1, 2, 10.0);
        var dataset = new Dataset<>(inputs, targets);
        var network = trainer.train(dataset);

        var badInput = factory.filled(1, 1, 1);
        assertThatThrownBy(() -> network.predict(badInput))
                .isInstanceOf(IllegalArgumentException.class);
    }

}