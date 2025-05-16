package org.jwcarman.netwerx.learning;

public class LearningRateProviders {

    private LearningRateProviders() {
        // prevent instantiation
    }


    /**
     * Constant learning rate provider.
     *
     * @param learningRate the learning rate
     * @return a learning rate provider that always returns the same learning rate
     */
    public static LearningRateProvider constant(double learningRate) {
        return _ -> learningRate;
    }

    /**
     * Exponential decay learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param decayRate           the decay rate
     * @return a learning rate provider that decreases the learning rate exponentially
     */
    public static LearningRateProvider exponentialDecay(double initialLearningRate, double decayRate) {
        return epoch -> initialLearningRate * Math.exp(-decayRate * epoch);
    }

    /**
     * Step decay learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param stepSize            the number of epochs after which to decay the learning rate
     * @param decayFactor         the factor by which to decay the learning rate
     * @return a learning rate provider that decreases the learning rate by a factor of decayFactor every stepSize epochs
     */
    public static LearningRateProvider stepDecay(double initialLearningRate, int stepSize, double decayFactor) {
        return epoch -> initialLearningRate * Math.pow(decayFactor, (epoch / stepSize));
    }

    /**
     * Polynomial decay learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param maxEpochs           the maximum number of epochs
     * @param power               the power to which to raise the decay factor
     * @return a learning rate provider that decreases the learning rate polynomially
     */
    public static LearningRateProvider polynomialDecay(double initialLearningRate, int maxEpochs, double power) {
        return epoch -> initialLearningRate * Math.pow(1 - (double) epoch / maxEpochs, power);
    }

    /**
     * Inverse time decay learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param decayRate           the decay rate
     * @return a learning rate provider that decreases the learning rate inversely with time
     */
    public static LearningRateProvider inverseTimeDecay(double initialLearningRate, double decayRate) {
        return epoch -> initialLearningRate / (1 + decayRate * epoch);
    }

    /**
     * Cosine annealing learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param maxEpochs           the maximum number of epochs
     * @return a learning rate provider that decreases the learning rate using cosine annealing
     */
    public static LearningRateProvider cosineAnnealing(double initialLearningRate, int maxEpochs) {
        return epoch -> {
            int clampedEpoch = Math.min(epoch, maxEpochs);
            return initialLearningRate * 0.5 * (1 + Math.cos(Math.PI * clampedEpoch / maxEpochs));
        };
    }

    /**
     * Cyclic learning rate provider.
     *
     * @param baseLearningRate the base learning rate
     * @param maxLearningRate  the maximum learning rate
     * @param stepSize         the number of epochs after which to switch the learning rate
     * @return a learning rate provider that oscillates between baseLearningRate and maxLearningRate
     */
    public static LearningRateProvider cyclicLearningRate(double baseLearningRate, double maxLearningRate, int stepSize) {
        return epoch -> {
            double cycle = 1 + (double) epoch / (2 * stepSize);
            double x = Math.abs((double) epoch / stepSize - 2 * cycle + 1);
            return baseLearningRate + (maxLearningRate - baseLearningRate) * Math.max(0, (1 - x));
        };
    }


    /**
     * Applies a warmup period to the given learning rate provider.
     *
     * @param base         the base learning rate provider
     * @param warmupEpochs the number of epochs to warm up
     * @param warmupTarget the target learning rate at the end of the warmup period
     * @return a learning rate provider that applies a warmup period to the base provider
     */
    public static LearningRateProvider withWarmup(LearningRateProvider base, int warmupEpochs, double warmupTarget) {
        return epoch -> epoch < warmupEpochs
                ? warmupTarget * epoch / warmupEpochs
                : base.getLearningRate(epoch);
    }

    /**
     * Warm restarts learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param cycleLength         the length of the cycle
     * @return a learning rate provider that decreases the learning rate using warm restarts
     */
    public static LearningRateProvider warmRestarts(double initialLearningRate, int cycleLength) {
        return epoch -> {
            double cycle = Math.floor(1 + (double) epoch / (2 * cycleLength));
            double x = Math.abs((double) epoch / cycleLength - 2 * cycle + 1);
            return initialLearningRate * Math.max(0, (1 - x));
        };
    }

    /**
     * Linear decay learning rate provider.
     *
     * @param initialLearningRate the initial learning rate
     * @param maxEpochs           the maximum number of epochs
     * @return a learning rate provider that decreases the learning rate linearly
     */
    public static LearningRateProvider linearDecay(double initialLearningRate, int maxEpochs) {
        return epoch -> initialLearningRate * Math.max(0, 1 - (double) epoch / maxEpochs);
    }


}
