package org.jwcarman.netwerx.learning;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.jwcarman.netwerx.util.Tolerances.withinTolerance;

class LearningRateProvidersTest {

    @Test
    void testConstantLearningRate() {
        double initialLearningRate = 0.01;
        var provider = LearningRateProviders.constant(initialLearningRate);
        for (int i = 0; i < 1_000_000; i+= 10000) {
            assertThat(provider.getLearningRate(i)).isEqualTo(initialLearningRate);
        }
    }

    @Test
    void testExponentialDecay() {
        double initialLearningRate = 0.1;
        double decayRate = 0.01;
        var provider = LearningRateProviders.exponentialDecay(initialLearningRate, decayRate);

        for (int i = 0; i < 100; i++) {
            double expected = initialLearningRate * Math.exp(-decayRate * i);
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testStepDecay() {
        double initialLearningRate = 0.1;
        int stepSize = 10;
        double decayFactor = 0.5;
        var provider = LearningRateProviders.stepDecay(initialLearningRate, stepSize, decayFactor);

        for (int i = 0; i < 100; i++) {
            double expected = initialLearningRate * Math.pow(decayFactor, (i / stepSize));
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testPolynomialDecay() {
        double initialLearningRate = 0.1;
        int maxEpochs = 100;
        double power = 2.0;
        var provider = LearningRateProviders.polynomialDecay(initialLearningRate, maxEpochs, power);

        for (int i = 0; i < maxEpochs; i++) {
            double expected = initialLearningRate * Math.pow(1 - (double) i / maxEpochs, power);
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testInverseTimeDecay() {
        double initialLearningRate = 0.1;
        double decayRate = 0.01;
        var provider = LearningRateProviders.inverseTimeDecay(initialLearningRate, decayRate);

        for (int i = 1; i < 100; i++) { // Start from 1 to avoid division by zero
            double expected = initialLearningRate / (1 + decayRate * i);
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testInverseTimeDecayWithZeroDecayRate() {
        double initialLearningRate = 0.1;
        double decayRate = 0.0;
        var provider = LearningRateProviders.inverseTimeDecay(initialLearningRate, decayRate);

        for (int i = 1; i < 100; i++) { // Start from 1 to avoid division by zero
            assertThat(provider.getLearningRate(i)).isEqualTo(initialLearningRate);
        }
    }

    @Test
    void testCosineAnnealing() {
        double initialLearningRate = 0.1;
        int maxEpochs = 100;
        var provider = LearningRateProviders.cosineAnnealing(initialLearningRate, maxEpochs);

        for (int i = 0; i < maxEpochs; i++) {
            double expected = initialLearningRate * 0.5 * (1 + Math.cos(Math.PI * i / maxEpochs));
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testCyclicLearningRate() {
        double baseLearningRate = 0.001;
        double maxLearningRate = 0.006;
        int stepSize = 2000;
        var provider = LearningRateProviders.cyclicLearningRate(baseLearningRate, maxLearningRate, stepSize);

        for (int i = 0; i < 10000; i++) {
            double cycle = 1 + (double) i / (2 * stepSize);
            double x = Math.abs(i / stepSize - 2 * cycle);
            double expected = baseLearningRate + (maxLearningRate - baseLearningRate) * Math.max(0, (1 - x));
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testCyclicLearningRateWithNegativeStepSize() {
        double baseLearningRate = 0.001;
        double maxLearningRate = 0.006;
        int stepSize = -2000; // This should be handled gracefully
        var provider = LearningRateProviders.cyclicLearningRate(baseLearningRate, maxLearningRate, stepSize);

        for (int i = 0; i < 10000; i++) {
            assertThat(provider.getLearningRate(i)).isEqualTo(baseLearningRate);
        }
    }

    @Test
    void testWithWarmup() {
        double initialLearningRate = 0.1;
        int warmupEpochs = 10;
        double warmupTarget = 0.5;
        var baseProvider = LearningRateProviders.constant(initialLearningRate);
        var provider = baseProvider.withWarmup(warmupEpochs, warmupTarget);

        for (int i = 0; i < warmupEpochs; i++) {
            double expected = warmupTarget * i / warmupEpochs;
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }

        for (int i = warmupEpochs; i < 100; i++) {
            assertThat(provider.getLearningRate(i)).isEqualTo(initialLearningRate);
        }
    }

    @Test
    void testWarmRestarts() {
        double initialLearningRate = 0.1;
        int cycleLength = 10;
        var provider = LearningRateProviders.warmRestarts(initialLearningRate, cycleLength);

        for (int i = 0; i < 100; i++) {
            double cycle = Math.floor(1 + (double) i / (2 * cycleLength));
            double x = Math.abs((double) i / cycleLength - 2 * cycle + 1);
            double expected = initialLearningRate * Math.max(0, (1 - x));
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }

    @Test
    void testLinearDecay() {
        double initialLearningRate = 0.1;
        int maxEpochs = 100;
        var provider = LearningRateProviders.linearDecay(initialLearningRate, maxEpochs);

        for (int i = 0; i < maxEpochs; i++) {
            double expected = initialLearningRate * Math.max(0, 1 - (double) i / maxEpochs);
            assertThat(provider.getLearningRate(i)).isCloseTo(expected, withinTolerance());
        }
    }
}