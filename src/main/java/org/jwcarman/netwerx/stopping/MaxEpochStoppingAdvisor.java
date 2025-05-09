package org.jwcarman.netwerx.stopping;

public class MaxEpochStoppingAdvisor implements StoppingAdvisor {

// ------------------------------ FIELDS ------------------------------

    public static final int DEFAULT_MAX_EPOCH = 100;
    private final int maxEpoch;

// --------------------------- CONSTRUCTORS ---------------------------

    public MaxEpochStoppingAdvisor() {
        this(DEFAULT_MAX_EPOCH);
    }

    public MaxEpochStoppingAdvisor(int maxEpoch) {
        this.maxEpoch = maxEpoch;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface StoppingAdvisor ---------------------

    @Override
    public boolean shouldStop(int epoch, double score) {
        return epoch >= maxEpoch;
    }

}
