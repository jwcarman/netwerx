package org.jwcarman.netwerx.mnist;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.activation.ActivationFunctions;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.observer.TrainingObservers;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regularization.Regularizations;
import org.jwcarman.netwerx.stopping.EpochCountStoppingAdvisor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.assertj.core.api.Assertions.assertThat;

class MnistTestCase {

    private final Logger logger = LoggerFactory.getLogger(MnistTestCase.class);

    @Test
    void testMnist() throws Exception {

        var factory = new EjmlMatrixFactory();
        var images = readImages(factory, 1000).rowSlice(0, 28 * 7);
        var input = images.columnSlice(0, 100);
        var validation = images.columnSlice(100, 110);
        var test = images.columnSlice(110, 160);
        var lossFunction = Losses.mse();
        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, images.rowCount())
                .stoppingAdvisor(new EpochCountStoppingAdvisor(500))
                .validationDataset(new Dataset<>(validation, validation))
                .defaultOptimizer(() -> Optimizers.adam(0.001, 0.9, 0.999, 1e-8))
                .denseLayer(layer -> layer.units(input.rowCount()).regularizationFunction(Regularizations.l2(1e-5)))
                .denseLayer(layer -> layer.units(32).activationFunction(ActivationFunctions.linear()))
                .denseLayer(layer -> layer
                        .units(input.rowCount())
                        .activationFunction(ActivationFunctions.sigmoid())
                        .regularizationFunction(Regularizations.l2(1e-5))
                )
                .lossFunction(lossFunction)
                .buildAutoencoderTrainer();

        var autoencoder = trainer.train(input, TrainingObservers.logging(logger, 100));

        var reconstructed = autoencoder.reconstruct(test);

        var loss = lossFunction.loss(reconstructed, test);
        assertThat(loss).isLessThanOrEqualTo(0.02);
    }

    private <M extends Matrix<M>> M readImages(MatrixFactory<M> factory, int imageCount) throws IOException {
        try (InputStream in = MnistTestCase.class.getResourceAsStream("/dataset/mnist/t10k-images-idx3-ubyte");
             DataInputStream din = new DataInputStream(in)) {
            din.readInt();
            var numberOfImages = din.readInt();
            var numberOfRows = din.readInt();
            var numberOfColumns = din.readInt();
            logger.info("Number of images: {}", numberOfImages);

            if (imageCount > numberOfImages) {
                throw new IllegalArgumentException("Number of images exceeds the maximum number of images");
            }

            var imageSize = numberOfRows * numberOfColumns;
            double[][] data = new double[imageSize][imageCount];
            for (int col = 0; col < imageCount; col++) {
                for (int row = 0; row < imageSize; row++) {
                    data[row][col] = din.readUnsignedByte() / 255.0;
                }
            }
            return factory.from(data);
        } finally {
            logger.info("Finished reading MNIST images.");
        }


    }
}
