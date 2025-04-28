package org.jwcarman.netwerx.titanic;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.NeuralNetworkBuilder;

import java.util.List;
import java.util.Random;

class TitanicTestCase {

// -------------------------- OTHER METHODS --------------------------

    @Test
    void testTitanicModel() throws Exception {
        var trainingPassengers = TitanicDatasetLoader.loadTrainingPassengers();

        var inputs = createInputs(trainingPassengers);
        var targets = new SimpleMatrix(trainingPassengers.stream().mapToDouble(p -> p.survived() ? 1.0 : 0.0).toArray()).transpose();

        System.out.println("Average target value: " + targets.elementSum() / targets.getNumCols());

        var rand = new Random(42);
        var classifier = new NeuralNetworkBuilder(inputs.getNumRows())
                .layer(layer -> layer
                        .units(8)
                        .random(rand)
                )
                .layer(layer -> layer
                        .units(4)
                        .random(rand)
                )
                .binaryClassifier(bc -> bc
                        .random(rand)
                );


        classifier.train(inputs, targets, (epoch, loss, a, y) -> epoch < 300);

        var testPassengers = TitanicDatasetLoader.loadTestPassengers();
        var testInputs = createInputs(testPassengers);
        var predictions = classifier.predict(testInputs);
        var survivors = 0;
        for (int col = 0; col < predictions.getNumCols(); col++) {
            if (predictions.get(0, col) >= 1.0) {
                survivors++;
            }
        }
        System.out.println("Predicted survivors: " + survivors + " out of " + predictions.getNumCols());
    }

    private static SimpleMatrix createInputs(List<TitanicPassenger> passengers) {
        final var data = new double[passengers.size()][6];
        for (int i = 0; i < passengers.size(); i++) {
            var passenger = passengers.get(i);
            data[i][0] = passenger.ticketClass();
            data[i][1] = passenger.age();
            data[i][2] = passenger.sex();
            data[i][3] = passenger.fare();
            data[i][4] = passenger.parentsAndChildren();
            data[i][5] = passenger.siblingsAndSpouses();
        }
        SimpleMatrix inputs = new SimpleMatrix(data).transpose();
        normalizeRow(inputs, 0);
        normalizeRow(inputs, 1);
        normalizeRow(inputs, 3);
        normalizeRow(inputs, 4);
        normalizeRow(inputs, 5);
        return inputs;
    }

    private static void normalizeRow(SimpleMatrix matrix, int rowIndex) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        // Find min and max in the specified row
        for (int col = 0; col < matrix.numCols(); col++) {
            double value = matrix.get(rowIndex, col);
            if (value < min) min = value;
            if (value > max) max = value;
        }

        double range = max - min;
        if (range == 0) {
            for (int col = 0; col < matrix.getNumCols(); col++) {
                matrix.set(rowIndex, col, 0.5);  // Uniform
            }
        } else {
            for (int col = 0; col < matrix.getNumCols(); col++) {
                double value = matrix.get(rowIndex, col);
                matrix.set(rowIndex, col, (value - min) / range);
            }
        }
    }

}
