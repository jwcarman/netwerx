package org.jwcarman.netwerx.wine;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.classification.multi.MultiClassifierStats;
import org.jwcarman.netwerx.data.CommaSeparatedValues;
import org.jwcarman.netwerx.data.Datasets;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.listener.TrainingListeners;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.network.DefaultNeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regularization.Regularizations;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class WineTestCase {

// ------------------------------ FIELDS ------------------------------

    private static final Logger logger = LoggerFactory.getLogger(WineTestCase.class);

    private static final Random random = new Random(4587985);

// -------------------------- OTHER METHODS --------------------------

    @Test
    void multiClassClassification() {
        final List<Wine> wines = CommaSeparatedValues.load("/dataset/wine/labeled.csv", csv -> {
            var label = Integer.parseInt(csv.get("class")) - 1;
            var alcohol = Double.parseDouble(csv.get("Alcohol"));
            var malicAcid = Double.parseDouble(csv.get("Malic acid"));
            var ash = Double.parseDouble(csv.get("Ash"));
            var alcalinityOfAsh = Double.parseDouble(csv.get("Alcalinity of ash"));
            var magnesium = Double.parseDouble(csv.get("Magnesium"));
            var totalPhenols = Double.parseDouble(csv.get("Total phenols"));
            var flavanoids = Double.parseDouble(csv.get("Flavanoids"));
            var nonFlavanoidPhenols = Double.parseDouble(csv.get("Nonflavanoid phenols"));
            var proanthocyanins = Double.parseDouble(csv.get("Proanthocyanins"));
            var colorIntensity = Double.parseDouble(csv.get("Color intensity"));
            var hue = Double.parseDouble(csv.get("Hue"));
            var od280Od315OfDilutedWines = Double.parseDouble(csv.get("OD280/OD315 of diluted wines"));
            var proline = Double.parseDouble(csv.get("Proline "));
            return new Wine(label, alcohol, malicAcid, ash, alcalinityOfAsh, magnesium, totalPhenols, flavanoids,
                    nonFlavanoidPhenols, proanthocyanins, colorIntensity, hue, od280Od315OfDilutedWines, proline);
        });

        var factory = new EjmlMatrixFactory();
        var split = Datasets.split(wines, 0.7f, 0.15f, 0.15f, random);

        var trainInputs = features(factory, split.train());
        var trainTargets = labels(split.train());

        var validationInputs = features(factory, split.validation());
        var validationTargets = validationInputs.multiClassifierClasses(3, labels(split.validation()));

        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(new EjmlMatrixFactory(), trainInputs.rowCount(), random)
                .defaultOptimizer(Optimizers::sgd)
                .validationDataset(new Dataset<>(validationInputs, validationTargets))
                //.stoppingAdvisor(new MaxEpochStoppingAdvisor(700))
                .listener(TrainingListeners.logging(logger, 100))
                .denseLayer(layer -> layer.units(16))
                .denseLayer(layer -> layer.units(8).regularizationFunction(Regularizations.l2(1e-4)))
                .buildMultiClassifierTrainer(3);

        var classifier = trainer.train(trainInputs, trainTargets);

        var testInputs = features(factory, split.test());
        var testTargets = labels(split.test());
        var predictions = classifier.predictClasses(testInputs);
        var stats = MultiClassifierStats.of(predictions, testTargets, 3);
        logger.info("Stats: {}", stats);
        assertThat(stats.f1()).isGreaterThanOrEqualTo(0.8);
    }

    private static <M extends Matrix<M>> M features(MatrixFactory<M> factory, List<Wine> list) {
        M features = Datasets.features(factory, list,
                Wine::alcohol,
                Wine::malicAcid,
                Wine::ash,
                Wine::alcalinityOfAsh,
                Wine::magnesium,
                Wine::totalPhenols,
                Wine::flavanoids,
                Wine::nonFlavanoidPhenols,
                Wine::proanthocyanins,
                Wine::colorIntensity,
                Wine::hue,
                Wine::od280Od315OfDilutedWines,
                Wine::proline);
        return features.normalizeRows();
    }

    private static int[] labels(List<Wine> list) {
        return Datasets.multiClassLabels(list, Wine::label);
    }

// -------------------------- INNER CLASSES --------------------------

    record Wine(int label,
                double alcohol,
                double malicAcid,
                double ash,
                double alcalinityOfAsh,
                double magnesium,
                double totalPhenols,
                double flavanoids,
                double nonFlavanoidPhenols,
                double proanthocyanins,
                double colorIntensity,
                double hue,
                double od280Od315OfDilutedWines,
                double proline) {

    }

}
