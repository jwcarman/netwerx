package org.jwcarman.netwerx.titanic;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierStats;
import org.jwcarman.netwerx.data.CommaSeparatedValues;
import org.jwcarman.netwerx.data.Datasets;
import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.def.DefaultNeuralNetworkTrainerBuilder;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.observer.TrainingObservers;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.stopping.EpochCountStoppingAdvisor;
import org.jwcarman.netwerx.util.Streams;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

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

        var split = Datasets.split(passengers, 0.7f, 0.15f, 0.15f, random);

        var factory = new EjmlMatrixFactory();

        var trainInputs = features(factory, split.train());
        var trainTargets = labels(split.train());
        var testInputs = features(factory, split.test());
        var testTargets = labels(split.test());
        var validationInputs = features(factory, split.validation());
        var validationTargets = factory.from(new double[][]{Streams.ofBooleans(labels(split.validation())).mapToDouble(l -> l ? 1.0 : 0.0).toArray()});


        var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, trainInputs.rowCount(), random)
                .validationDataset(new Dataset<>(validationInputs, validationTargets))
                .defaultOptimizer(Optimizers::sgd)
                .stoppingAdvisor(new EpochCountStoppingAdvisor(200))
                .denseLayer(layer -> layer.units(8)
                        .weightOptimizer(Optimizers::sgd)
                        .biasOptimizer(Optimizers::sgd))
                .denseLayer(layer -> layer.units(4))
                .buildBinaryClassifierTrainer();

        var classifier = trainer.train(trainInputs, trainTargets, TrainingObservers.logging(logger, 100));


        var predictions = classifier.predict(testInputs);
        var stats = BinaryClassifierStats.of(predictions, testTargets);
        logger.info("Stats: {}", stats);
        assertThat(stats.f1()).isGreaterThanOrEqualTo(0.7);
    }

    private static boolean[] labels(List<TitanicPassenger> list) {
        return Datasets.binaryLabels(list, TitanicPassenger::survived);
    }

    private static <M extends Matrix<M>> M features(MatrixFactory<M> factory, List<TitanicPassenger> list) {
        M features = Datasets.features(factory, list,
                TitanicPassenger::ticketClass,
                TitanicPassenger::age,
                TitanicPassenger::sex,
                TitanicPassenger::fare,
                TitanicPassenger::parentsAndChildren,
                TitanicPassenger::siblingsAndSpouses);
        return features
                .normalizeRow(0)
                .normalizeRow(1)
                .normalizeRow(3)
                .normalizeRow(4)
                .normalizeRow(5);
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
