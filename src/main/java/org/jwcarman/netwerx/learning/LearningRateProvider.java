package org.jwcarman.netwerx.learning;

@FunctionalInterface
public interface LearningRateProvider {
    double getLearningRate(int epoch);

    default LearningRateProvider withWarmup(int warmupEpochs, double warmupTarget) {
        return LearningRateProviders.withWarmup(this, warmupEpochs, warmupTarget);
    }
}
