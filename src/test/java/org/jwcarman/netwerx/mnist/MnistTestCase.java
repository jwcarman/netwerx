package org.jwcarman.netwerx.mnist;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.activation.ActivationFunctions;
import org.jwcarman.netwerx.batch.TrainingExecutors;
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
import org.jwcarman.netwerx.util.Randoms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import java.util.concurrent.Executors;

import static org.assertj.core.api.Assertions.assertThat;

class MnistTestCase {

    private final Logger logger = LoggerFactory.getLogger(MnistTestCase.class);

    @Test
    void testMnist() throws Exception {
        var random = new Random();
        var factory = new EjmlMatrixFactory();
        var images = readImages(factory, 320).rowSlice(0, 28 * 7);
        var dataset = new Dataset<>(images, images);
        var split = dataset.split(random, 0.8);
        var input = split.left();
        split = split.right().split(random, 0.5);
        var validation = split.left();
        var test = split.right();

        var lossFunction = Losses.mse();

        logger.info("Training on {} images.", input.features().columnCount());
        logger.info("Validation on {} images.", validation.features().columnCount());
        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, images.rowCount(), random)
                .trainingExecutor(TrainingExecutors.miniBatch(32, Randoms.defaultRandom(), Executors.newFixedThreadPool(10)))
                .stoppingAdvisor(new EpochCountStoppingAdvisor(500))
                .validationDataset(validation)
                .defaultOptimizer(() -> Optimizers.adam(0.001, 0.9, 0.999, 1e-8))
                .denseLayer(layer -> layer.units(input.features().rowCount()).regularizationFunction(Regularizations.l2(1e-5)))
                .denseLayer(layer -> layer.units(32).activationFunction(ActivationFunctions.linear()))
                .denseLayer(layer -> layer
                        .units(input.features().rowCount())
                        .activationFunction(ActivationFunctions.sigmoid())
                        .regularizationFunction(Regularizations.l2(1e-5))
                )
                .lossFunction(Losses.mse())
                .buildAutoencoderTrainer();
        var before = System.nanoTime();
        var autoencoder = trainer.train(input.features(), TrainingObservers.logging(logger, 100));
        var after = System.nanoTime();

        logger.info("Training took {} ms", (after - before) / 1_000_000);

        logger.info("Testing on {} images.", test.features().columnCount());
        var reconstructed = autoencoder.reconstruct(test.features());

        var loss = lossFunction.loss(reconstructed, test.features());
        logger.info("Testing loss: {}", loss);
        assertThat(loss).isLessThanOrEqualTo(0.02);
    }

    private <M extends Matrix<M>> M readImages(MatrixFactory<M> factory, int imageCount) throws IOException {
        var before = System.nanoTime();
        try (InputStream in = MnistTestCase.class.getResourceAsStream("/dataset/mnist/t10k-images-idx3-ubyte");
             DataInputStream din = new DataInputStream(in)) {
            din.readInt();
            var numberOfImages = din.readInt();
            var numberOfRows = din.readInt();
            var numberOfColumns = din.readInt();

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
            var after = System.nanoTime();
            logger.info("Finished reading {} MNIST images in {} ms", imageCount, (after - before) / 1_000_000);
        }


    }
}
