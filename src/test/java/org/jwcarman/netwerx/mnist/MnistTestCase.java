package org.jwcarman.netwerx.mnist;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.activation.ActivationFunctions;
import org.jwcarman.netwerx.classification.multi.MultiClassifierStats;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.learning.LearningRateProviders;
import org.jwcarman.netwerx.listener.TrainingListeners;
import org.jwcarman.netwerx.loss.LossFunctions;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regularization.Regularizations;
import org.jwcarman.netwerx.score.ScoringFunctions;
import org.jwcarman.netwerx.stopping.StoppingAdvisors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class MnistTestCase {
    public static final int MC_IMAGE_COUNT = 60000;
    public static final int MC_VALIDATION_IMAGE_COUNT = 2000;

// ------------------------------ FIELDS ------------------------------

    private final Logger logger = LoggerFactory.getLogger(MnistTestCase.class);

// -------------------------- OTHER METHODS --------------------------

    @Test
    void mnistAutoencoder() {
        var random = new Random(11223344);
        var factory = new EjmlMatrixFactory();
        var images = MnistReader.readTrainingImages(320, factory).rowSlice(0, 28 * 7);
        var dataset = new Dataset<>(images, images);
        var split = dataset.split(random, 0.8);
        var training = split.left();
        split = split.right().split(random, 0.5);
        var validation = split.left();
        var test = split.right();

        var lossFunction = LossFunctions.mse();

        logger.info("Training on {} images.", training.features().columnCount());
        logger.info("Validation on {} images.", validation.features().columnCount());

        var learningRateProvider = LearningRateProviders.constant(0.001);

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, images.rowCount(), random)
                .stoppingAdvisor(StoppingAdvisors.scoreThreshold(-0.02))
                .scoringFunction(ScoringFunctions.validationLoss())
                .validationDataset(validation)
                .subBatchCount(4)
                .listener(TrainingListeners.logging(logger, 100))
                .defaultOptimizer(() -> Optimizers.adam(learningRateProvider, 0.9, 0.999, 1e-8))
                .denseLayer(layer -> layer.units(training.features().rowCount()).regularizationFunction(Regularizations.l2(1e-5)))
                .dropoutLayer(layer -> layer.dropoutRate(0.45))
                .denseLayer(layer -> layer.units(32).activationFunction(ActivationFunctions.linear()))
                .dropoutLayer()
                .denseLayer(layer -> layer
                        .units(training.features().rowCount())
                        .activationFunction(ActivationFunctions.sigmoid())
                        .regularizationFunction(Regularizations.l2(1e-5))
                )
                .buildAutoencoderTrainer();
        var before = System.nanoTime();
        var autoencoder = trainer.train(training.features());
        var after = System.nanoTime();

        logger.info("Training took {} ms", (after - before) / 1_000_000);

        logger.info("Testing on {} images.", test.features().columnCount());
        var reconstructed = autoencoder.reconstruct(test.features());

        var loss = lossFunction.loss(reconstructed, test.features());
        logger.info("Testing loss: {}", loss);
        assertThat(loss).isLessThanOrEqualTo(0.02);
    }

    //@Test
    void mnistMultiClassifier() {
        var random = new Random(42);
        var factory = new EjmlMatrixFactory();
        var images = MnistReader.readTrainingImages(MC_IMAGE_COUNT, factory);
        var labels = MnistReader.readTrainingLabels(MC_IMAGE_COUNT);

        var trainingSize = MC_IMAGE_COUNT - MC_VALIDATION_IMAGE_COUNT;
        var trainingImages = images.columnSlice(0, trainingSize);
        var trainingLabels = Arrays.copyOfRange(labels, 0, trainingSize);

        var validationImages = images.columnSlice(trainingSize, MC_IMAGE_COUNT);
        var validationLabels = Arrays.copyOfRange(labels, trainingSize, MC_IMAGE_COUNT);
        var validationDataset = Dataset.forMultiClassifier(validationImages, 10, validationLabels);

        var testImages = MnistReader.readTestImages(factory);
        var testLabels = MnistReader.readTestLabels();

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, images.rowCount(), random)
                .batchSize(1024)
                .validationDataset(validationDataset)
                .defaultOptimizer(Optimizers::adam)
                .listener(TrainingListeners.logging(logger, 10))
                .scoringFunction(ScoringFunctions.validationLoss())
                .stoppingAdvisor(StoppingAdvisors.patience())
                .denseLayer(layer -> layer.units(64).regularizationFunction(Regularizations.l2(1e-4)))
                .denseLayer(layer -> layer.units(32).regularizationFunction(Regularizations.l2(1e-4)))
                .denseLayer(layer -> layer.units(16).regularizationFunction(Regularizations.l2(1e-4)))
                .buildMultiClassifierTrainer(10);


        var network = trainer.train(trainingImages, trainingLabels);


        var predictions = network.predictClasses(testImages);
        var stats = MultiClassifierStats.of(predictions, testLabels, 10);
        logger.info("Stats: {}", stats);

    }

}
