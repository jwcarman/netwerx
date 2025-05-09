package org.jwcarman.netwerx.stopping;

import java.util.Arrays;

public class StoppingAdvisors {

// -------------------------- STATIC METHODS --------------------------


    private StoppingAdvisors() {
        // Prevent instantiation
    }

    public static StoppingAdvisor maxEpoch(int maxEpoch) {
        return new MaxEpochStoppingAdvisor(maxEpoch);
    }

    public static StoppingAdvisor maxEpoch() {
        return new MaxEpochStoppingAdvisor();
    }

    public static StoppingAdvisor patience() {
        return new PatienceStoppingAdvisor();
    }

    public static StoppingAdvisor patience(int patience, double minDelta) {
        return new PatienceStoppingAdvisor(patience, minDelta);
    }

    public static StoppingAdvisor scoreThreshold() {
        return new ScoreThresholdStoppingAdvisor();
    }

    public static StoppingAdvisor scoreThreshold(double threshold) {
        return new ScoreThresholdStoppingAdvisor(threshold);
    }


    public static StoppingAdvisor composite(StoppingAdvisor... advisors) {
        return ((epoch, score) -> Arrays.stream(advisors).anyMatch(a -> a.shouldStop(epoch, score)));
    }

}
