package org.jwcarman.netwerx.autoencoder;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DefaultAutoencoderTrainerTest {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testFactoryMethod() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .denseLayer(layer -> layer.units(10))
                .denseLayer(layer -> layer.units(5))
                .denseLayer(layer -> layer.units(10))
                .buildAutoencoderTrainer();

        assertThat(trainer).isNotNull();
    }

    @Test
    void testFactoryMethodFailsWithLessThanThreeLayers() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .denseLayer(layer -> layer.units(10))
                .denseLayer(layer -> layer.units(10));
        assertThatThrownBy(trainer::buildAutoencoderTrainer)
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void testFactoryMethodFailsWithMismatchedOutput() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 10)
                .denseLayer(layer -> layer.units(10))
                .denseLayer(layer -> layer.units(4))
                .denseLayer(layer -> layer.units(6));
        assertThatThrownBy(trainer::buildAutoencoderTrainer)
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void testTrainingCreatesAutoencoderInstance() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 5)
                .denseLayer(layer -> layer.units(5))
                .denseLayer(layer -> layer.units(2))
                .denseLayer(layer -> layer.units(5))
                .buildAutoencoderTrainer();

        var input = factory.from(5, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        var autoencoder = trainer.train(input);

        assertThat(autoencoder).isNotNull();
    }

    @Test
    void testEncode() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 5)
                .denseLayer(layer -> layer.units(5))
                .denseLayer(layer -> layer.units(2))
                .denseLayer(layer -> layer.units(5))
                .buildAutoencoderTrainer();

        var input = factory.from(5, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        var autoencoder = trainer.train(input);

        var encoded = autoencoder.encode(input);
        assertThat(encoded.rowCount()).isEqualTo(2);
    }

    @Test
    void testDecode() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 5)
                .denseLayer(layer -> layer.units(5))
                .denseLayer(layer -> layer.units(2))
                .denseLayer(layer -> layer.units(5))
                .buildAutoencoderTrainer();

        var input = factory.from(5, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        var autoencoder = trainer.train(input);

        var encoded = autoencoder.encode(input);
        var decoded = autoencoder.decode(encoded);
        assertThat(decoded.rowCount()).isEqualTo(5);
    }

    @Test
    void testReconstruct() {
        var factory = new EjmlMatrixFactory();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 5)
                .denseLayer(layer -> layer.units(5))
                .denseLayer(layer -> layer.units(2))
                .denseLayer(layer -> layer.units(5))
                .buildAutoencoderTrainer();

        var input = factory.from(5, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        var autoencoder = trainer.train(input);
        var reconstructed = autoencoder.reconstruct(input);
        assertThat(reconstructed.rowCount()).isEqualTo(5);
    }

}