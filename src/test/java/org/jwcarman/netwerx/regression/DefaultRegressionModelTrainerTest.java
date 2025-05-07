package org.jwcarman.netwerx.regression;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DefaultRegressionModelTrainerTest {

    @Test
    void testTrainingWithBadLabelsThrowsException() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 5)
                .denseLayer()
                .buildRegressionModelTrainer();

        var inputs = factory.filled(2, 2, 1.0);
        var labels = new double[]{1.0, 2.0, 3.0};
        assertThatThrownBy(() -> trainer.train(inputs, labels))
                .isInstanceOf(IllegalArgumentException.class);

    }

    @Test
    void shouldTrainWithNoObserver() {
        var factory = new EjmlMatrixFactory();
        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .denseLayer()
                .denseLayer()
                .buildRegressionModelTrainer();

        var inputs = factory.filled(10, 1, 5.0);
        var labels = new double[] {2990.0};

        var network = trainer.train(inputs, labels);
        assertThat(network).isNotNull();
    }

}