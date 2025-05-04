package org.jwcarman.netwerx.concrete;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.data.CommaSeparatedValues;
import org.jwcarman.netwerx.data.Datasets;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.observer.TrainingObservers;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regression.RegressionModelStats;
import org.jwcarman.netwerx.regularization.Regularizations;
import org.jwcarman.netwerx.stopping.EpochCountStoppingAdvisor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class ConcreteTestCase {

    private static final Logger logger = LoggerFactory.getLogger(ConcreteTestCase.class);

    private static final Random random = new Random(42);

    @Test
    void regressionModel() {
        List<Concrete> concretes = CommaSeparatedValues.load("/dataset/concrete/labeled.csv", csv -> {
            var cement = Double.parseDouble(csv.get("Cement"));
            var blastFurnaceSlag = Double.parseDouble(csv.get("Blast Furnace Slag"));
            var flyAsh = Double.parseDouble(csv.get("Fly Ash"));
            var water = Double.parseDouble(csv.get("Water"));
            var superplasticizer = Double.parseDouble(csv.get("Superplasticizer"));
            var coarseAggregate = Double.parseDouble(csv.get("Coarse Aggregate"));
            var fineAggregate = Double.parseDouble(csv.get("Fine Aggregate"));
            var age = Double.parseDouble(csv.get("Age"));
            var strength = Double.parseDouble(csv.get("Strength"));
            return new Concrete(cement, blastFurnaceSlag, flyAsh, water, superplasticizer, coarseAggregate, fineAggregate, age, strength);
        });

        var factory = new EjmlMatrixFactory();

        var split = Datasets.split(concretes, 0.8f, 0.1f, 0.1f, random);
        var trainInputs = features(factory, split.train());
        var trainTargets = labels(split.train());
        var validationInputs = features(factory, split.validation());
        var validationTargets = factory.from(1, validationInputs.columnCount(), labels(split.validation()));

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, trainInputs.rowCount(), random)
                .defaultOptimizer(() -> Optimizers.momentum(0.25, 0.9))
                .validationDataset(new Dataset<>(validationInputs, validationTargets))
                .stoppingAdvisor(new EpochCountStoppingAdvisor(400))
                .denseLayer(layer -> layer.units(16))
                .denseLayer(layer -> layer.units(4))
                .denseLayer(layer -> layer.units(4).regularizationFunction(Regularizations.l2(1e-4)))
                .buildRegressionModelTrainer();

        var regressionModel = trainer.train(trainInputs, trainTargets, TrainingObservers.logging(logger, 100));

        var testInputs = features(factory, split.test());
        var testTargets = labels(split.test());
        var predictions = regressionModel.predict(testInputs);

        var stats = RegressionModelStats.of(predictions, testTargets);
        logger.info("Stats: {}", stats);
        assertThat(stats.r2()).isGreaterThan(0.8);

    }

    private static double[] labels(List<Concrete> list) {
        return Datasets.regressionLabels(list, Concrete::strength);
    }

    private static <M extends Matrix<M>> M features(MatrixFactory<M> factory, List<Concrete> list) {
        M features = Datasets.features(factory, list,
                Concrete::cement,
                Concrete::blastFurnaceSlag,
                Concrete::flyAsh,
                Concrete::water,
                Concrete::superplasticizer,
                Concrete::coarseAggregate,
                Concrete::fineAggregate,
                Concrete::age);

        return features.normalizeRows();
    }

    // Cement	Blast Furnace Slag	Fly Ash	Water	Superplasticizer	Coarse Aggregate	Fine Aggregate	Age	Strength
    record Concrete(
            double cement,
            double blastFurnaceSlag,
            double flyAsh,
            double water,
            double superplasticizer,
            double coarseAggregate,
            double fineAggregate,
            double age,
            double strength
    ) {

    }
}
