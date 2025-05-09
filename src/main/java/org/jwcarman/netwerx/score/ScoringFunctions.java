package org.jwcarman.netwerx.score;

public class ScoringFunctions {

// -------------------------- STATIC METHODS --------------------------

    public static ScoringFunction validationLoss() {
        return outcome -> -outcome.validationLoss(); // lower loss = higher score
    }

    public static ScoringFunction validationLossWithPenalty() {
        return outcome -> -(outcome.validationLoss() + outcome.regularizationPenalty());
    }

    public static ScoringFunction trainingLoss() {
        return outcome -> -outcome.trainingLoss();
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private ScoringFunctions() {
        // Prevent instantiation
    }

}
