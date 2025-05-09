package org.jwcarman.netwerx.stopping;

public class PatienceStoppingAdvisor implements StoppingAdvisor {

// ------------------------------ FIELDS ------------------------------

    public static final int DEFAULT_PATIENCE = 10;
    public static final double DEFAULT_MIN_DELTA = 1e-4;
    private final int patience;
    private final double minDelta;
    private int bestEpoch = 0;
    private double bestScore = Double.NEGATIVE_INFINITY;

// --------------------------- CONSTRUCTORS ---------------------------

    public PatienceStoppingAdvisor() {
        this(DEFAULT_PATIENCE, DEFAULT_MIN_DELTA);
    }

    public PatienceStoppingAdvisor(int patience, double minDelta) {
        this.patience = patience;
        this.minDelta = minDelta;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface StoppingAdvisor ---------------------

    @Override
    public boolean shouldStop(int epoch, double score) {
        if (score > bestScore + minDelta) {
            bestScore = score;
            bestEpoch = epoch;
        }
        return epoch - bestEpoch >= patience;
    }

}
