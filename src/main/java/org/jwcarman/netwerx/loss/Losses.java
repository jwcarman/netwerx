package org.jwcarman.netwerx.loss;

public class Losses {

// -------------------------- STATIC METHODS --------------------------

    public static Loss bce() {
        return new BinaryCrossEntropy();
    }

    public static Loss bce(double epsilon) {
        return new BinaryCrossEntropy(epsilon);
    }

    public static Loss mse() {
        return new MeanSquaredError();
    }

    public static Loss cce() {
        return new CategoricalCrossEntropy();
    }

    public static Loss cce(double epsilon) {
        return new CategoricalCrossEntropy(epsilon);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Losses() {
        // Prevent instantiation
    }

}
