package org.jwcarman.netwerx.loss;

public class LossFunctions {

// -------------------------- STATIC METHODS --------------------------

    public static LossFunction bce() {
        return new BinaryCrossEntropy();
    }

    public static LossFunction bce(double epsilon) {
        return new BinaryCrossEntropy(epsilon);
    }

    public static LossFunction mse() {
        return new MeanSquaredError();
    }

    public static LossFunction cce() {
        return new CategoricalCrossEntropy();
    }

    public static LossFunction cce(double epsilon) {
        return new CategoricalCrossEntropy(epsilon);
    }

    public static LossFunction mae() {
        return new MeanAbsoluteError();
    }

    public static LossFunction huber() {
        return new Huber();
    }

    public static LossFunction huber(double delta) {
        return new Huber(delta);
    }

    public static LossFunction hinge() {
        return new Hinge();
    }

    public static LossFunction logCosh() {
        return new LogCosh();
    }

    public static LossFunction weightedBce(double positiveWeight, double negativeWeight) {
        return new WeightedBinaryCrossEntropy(positiveWeight, negativeWeight);
    }

    public static LossFunction weightedBce(double positiveWeight, double negativeWeight, double epsilon) {
        return new WeightedBinaryCrossEntropy(positiveWeight, negativeWeight, epsilon);
    }

    // --------------------------- CONSTRUCTORS ---------------------------

    private LossFunctions() {
        // Prevent instantiation
    }

}
