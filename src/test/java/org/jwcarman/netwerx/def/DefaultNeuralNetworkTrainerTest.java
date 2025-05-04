package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultNeuralNetworkTrainerTest {

    @Test
    void emptyValidationSetShouldYieldNaN() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 2)
                .denseLayer()
                .denseLayer(layer -> layer.units(1))
                .build();

        var inputs = factory.filled(2, 2, 1.0);
        var targets = factory.filled(1, 2, 10.0);
        var dataset = new Dataset<>(inputs, targets);
        trainer.train(dataset, outcome -> assertThat(outcome.validationLoss()).isNaN());
    }

}