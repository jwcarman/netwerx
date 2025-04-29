package org.jwcarman.netwerx.wine;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.classification.multi.MultiClassifierStats;
import org.jwcarman.netwerx.data.CommaSeparatedValues;
import org.jwcarman.netwerx.data.Datasets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.data.Datasets.normalizeFeature;

class WineTestCase {

    private static final Logger logger = LoggerFactory.getLogger(WineTestCase.class);

    private static final Random random = new Random(42);

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

        var split = Datasets.split(wines, 0.8f, random);

        var trainInputs = features(split.trainingSet());
        var trainTargets = labels(split.trainingSet());

        var classifier = NeuralNetwork.builder(trainInputs.getNumRows())
                .layer(layer -> layer
                        .units(16)
                        .random(random))
                .layer(layer -> layer
                        .units(8)
                        .random(random))
                .multiClassifier(mc -> mc.outputClasses(3)
                        .random(random));

        classifier.train(trainInputs, trainTargets, (epoch, loss, a, y) -> epoch < 600);

        var testInputs = features(split.testSet());
        var testTargets = labels(split.testSet());
        var predictions = classifier.predict(testInputs);
        var stats = MultiClassifierStats.of(predictions, testTargets, 3);
        logger.info("Stats: {}", stats);
        assertThat(stats.f1()).isGreaterThanOrEqualTo(0.75);

    }

    private static int[] labels(List<Wine> list) {
        return Datasets.multiClassLabels(list, Wine::label);
    }

    private static SimpleMatrix features(List<Wine> list) {
        SimpleMatrix features = Datasets.features(list,
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
        normalizeFeature(features, 0); // Alcohol
        normalizeFeature(features, 1); // Malic Acid
        normalizeFeature(features, 2); // Ash
        normalizeFeature(features, 3); // Alcalinity of Ash
        normalizeFeature(features, 4); // Magnesium
        normalizeFeature(features, 5); // Total Phenols
        normalizeFeature(features, 6); // Flavanoids
        normalizeFeature(features, 7); // Non-Flavanoid Phenols
        normalizeFeature(features, 8); // Proanthocyanins
        normalizeFeature(features, 9); // Color Intensity
        normalizeFeature(features, 10); // Hue
        normalizeFeature(features, 11); // OD280/OD315 of Diluted Wines
        normalizeFeature(features, 12); // Proline
        return features;
    }

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
