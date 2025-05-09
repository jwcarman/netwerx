package org.jwcarman.netwerx.stopping;

public class ScoreThresholdStoppingAdvisor implements StoppingAdvisor {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_THRESHOLD = -0.01;
    private final double threshold;

// --------------------------- CONSTRUCTORS ---------------------------

    public ScoreThresholdStoppingAdvisor() {
        this(DEFAULT_THRESHOLD);
    }

    public ScoreThresholdStoppingAdvisor(double threshold) {
        this.threshold = threshold;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface StoppingAdvisor ---------------------

    @Override
    public boolean shouldStop(int epoch, double score) {
        return score >= threshold;
    }

}
