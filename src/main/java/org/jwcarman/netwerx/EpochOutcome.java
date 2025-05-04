package org.jwcarman.netwerx;

public record EpochOutcome(int epoch, double trainingLoss, double validationLoss, double regularizationPenalty, double totalLoss) {
}
