package org.jwcarman.netwerx.classification.binary;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.def.DefaultNeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DefaultBinaryClassifierTrainerTest {

    @Test
    void shouldRejectInvalidTrainingDataset() {
        var factory = new EjmlMatrixFactory();
        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .buildBinaryClassifierTrainer();

        var inputs = factory.filled(10, 10, 5.0);
        var labels = new boolean[]{false, false};

        assertThatThrownBy(() -> trainer.train(inputs, labels))
                .isInstanceOf(IllegalArgumentException.class);
    }

}