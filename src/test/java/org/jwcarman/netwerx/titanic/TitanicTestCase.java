package org.jwcarman.netwerx.titanic;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.TrainingObserver;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierStats;
import org.jwcarman.netwerx.data.CommaSeparatedValues;
import org.jwcarman.netwerx.data.Datasets;
import org.jwcarman.netwerx.def.DefaultNeuralNetworkBuilder;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.optimization.OptimizerProvider;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.data.Datasets.normalizeFeature;

class TitanicTestCase {

    private static final Logger logger = LoggerFactory.getLogger(TitanicTestCase.class);

    public static final String AGE = "Age";
    public static final String FARE = "Fare";
    public static final double DEFAULT_AGE = 30.0;
    public static final String PARENTS_AND_CHILDREN = "Parch";
    public static final String SIBLINGS_AND_SPOUSES = "SibSp";
    public static final String TICKET_CLASS = "Pclass";
    public static final String MALE = "male";
    public static final String SEX = "Sex";
    public static final String SURVIVED = "Survived";
    public static final String TRUE = "1";
    public static final String NAME = "Name";

    private final Random random = new Random(42);

// -------------------------- OTHER METHODS --------------------------

    @Test
    void binaryClassifier() {
        final List<TitanicPassenger> passengers = CommaSeparatedValues.load("/dataset/titanic/labeled.csv", csv -> {
            var name = csv.get(NAME);
            var fare = csv.get(FARE).isEmpty() ? 32.0 : Double.parseDouble(csv.get(FARE));
            var ticketClass = Integer.parseInt(csv.get(TICKET_CLASS));
            var age = csv.get(AGE).isEmpty() ? DEFAULT_AGE : Double.parseDouble(csv.get(AGE));
            var parentsAndChildren = csv.get(PARENTS_AND_CHILDREN).isEmpty() ? 0 : Integer.parseInt(csv.get(PARENTS_AND_CHILDREN));
            var siblingsAndSpouses = csv.get(SIBLINGS_AND_SPOUSES).isEmpty() ? 0 : Integer.parseInt(csv.get(SIBLINGS_AND_SPOUSES));
            var sex = MALE.equalsIgnoreCase(csv.get(SEX)) ? 1 : 0;
            var survived = csv.isSet(SURVIVED) && TRUE.equals(csv.get(SURVIVED));
            return new TitanicPassenger(name, ticketClass, age, sex, fare, parentsAndChildren, siblingsAndSpouses, survived);
        });

        var split = Datasets.split(passengers, 0.8f, random);

        var trainInputs = features(split.trainingSet());
        var trainTargets = labels(split.trainingSet());

        var classifier = new DefaultNeuralNetworkBuilder<>(new EjmlMatrixFactory(), trainInputs.rowCount())
                .random(random)
                .layer(layer -> layer.units(8))
                .layer(layer -> layer.units(4))
                .binaryClassifier();

        OptimizerProvider<EjmlMatrix> provider = Optimizers.uniform(Optimizers::sgd);

        classifier.train(trainInputs, trainTargets, provider, new TrainingObserver() {
            @Override
            public <M extends Matrix<M>> boolean onEpoch(int epoch, double loss, M yHat, M y) {
                return epoch < 100;
            }
        });

        var testInputs = features(split.testSet());
        var testTargets = labels(split.testSet());
        var predictions = classifier.predict(testInputs);
        var stats = BinaryClassifierStats.of(predictions, testTargets);
        logger.info("Stats: {}", stats);
        assertThat(stats.f1()).isGreaterThanOrEqualTo(0.7);
    }

    private static boolean[] labels(List<TitanicPassenger> list) {
        return Datasets.binaryLabels(list, TitanicPassenger::survived);
    }

    private static EjmlMatrix features(List<TitanicPassenger> list) {
        EjmlMatrix features = Datasets.features(list,
                TitanicPassenger::ticketClass,
                TitanicPassenger::age,
                TitanicPassenger::sex,
                TitanicPassenger::fare,
                TitanicPassenger::parentsAndChildren,
                TitanicPassenger::siblingsAndSpouses);
        normalizeFeature(features, 0); // Ticket Class
        normalizeFeature(features, 1); // Age
        normalizeFeature(features, 3); // Fare
        normalizeFeature(features, 4); // Parents and Children
        normalizeFeature(features, 5); // Siblings and Spouses
        return features;
    }

    record TitanicPassenger(String name,
                            double ticketClass,
                            double age,
                            double sex,
                            double fare,
                            double siblingsAndSpouses,
                            double parentsAndChildren,
                            boolean survived) {

    }
}
