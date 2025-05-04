package org.jwcarman.netwerx.classification.multi;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DefaultMultiClassifierTrainerTest {
    @Test
    void shouldRejectInvalidTrainingDataset() {
        var factory = new EjmlMatrixFactory();
        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .buildMultiClassifierTrainer(5);
        var inputs = factory.filled(10, 10, 5.0);
        var targets = new int[]{1, 2};
        assertThatThrownBy(() -> trainer.train(inputs, targets))
                .isInstanceOf(IllegalArgumentException.class);
    }
}