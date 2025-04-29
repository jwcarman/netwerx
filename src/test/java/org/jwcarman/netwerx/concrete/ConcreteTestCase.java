package org.jwcarman.netwerx.concrete;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.data.CommaSeparatedValues;
import org.jwcarman.netwerx.data.Datasets;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regression.RegressionModelStats;
import org.jwcarman.netwerx.util.Customizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.data.Datasets.normalizeFeature;

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

        var split = Datasets.split(concretes, 0.8f, random);
        var trainInputs = features(split.trainingSet());
        var trainTargets = labels(split.trainingSet());

        var classifier = NeuralNetwork.builder(trainInputs.getNumRows())
                .random(random)
                .optimizer(ConcreteTestCase::optimizer)
                .layer(layer -> layer
                        .units(16)
                )
                .layer(layer -> layer
                        .units(8)
                )
                .regressionModel();

        classifier.train(trainInputs, trainTargets, (epoch, loss, a, y) -> epoch < 2000);
        var testInputs = features(split.testSet());
        var testTargets = labels(split.testSet());
        var predictions = classifier.predict(testInputs);

        var stats = RegressionModelStats.of(predictions, testTargets);
        logger.info("Stats: {}", stats);
        assertThat(stats.r2()).isGreaterThan(0.8);

    }

    private static Optimizer optimizer() {
        return Optimizers.momentum(0.25, 0.9);
    }

    private static double[] labels(List<Concrete> list) {
        return Datasets.regressionLabels(list, Concrete::strength);
    }

    private static SimpleMatrix features(List<Concrete> list) {
        SimpleMatrix features = Datasets.features(list,
                Concrete::cement,
                Concrete::blastFurnaceSlag,
                Concrete::flyAsh,
                Concrete::water,
                Concrete::superplasticizer,
                Concrete::coarseAggregate,
                Concrete::fineAggregate,
                Concrete::age);

        normalizeFeature(features, 0); // Cement
        normalizeFeature(features, 1); // Blast Furnace Slag
        normalizeFeature(features, 2); // Fly Ash
        normalizeFeature(features, 3); // Water
        normalizeFeature(features, 4); // Superplasticizer
        normalizeFeature(features, 5); // Coarse Aggregate
        normalizeFeature(features, 6); // Fine Aggregate
        normalizeFeature(features, 7); // Age
        return features;
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
