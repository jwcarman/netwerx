package org.jwcarman.netwerx.network;

public record EpochOutcome(int epoch, double trainingLoss, double validationLoss, double regularizationPenalty, double totalLoss) {
}
