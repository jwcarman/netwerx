package org.jwcarman.netwerx.loss;

public class Losses {

// -------------------------- STATIC METHODS --------------------------

    public static LossFunction bce() {
        return new BinaryCrossEntropy();
    }

    public static LossFunction bce(double epsilon) {
        return new BinaryCrossEntropy(epsilon);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Losses() {
        // Prevent instantiation
    }

}
