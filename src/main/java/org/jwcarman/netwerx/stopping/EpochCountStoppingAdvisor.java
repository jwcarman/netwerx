package org.jwcarman.netwerx.stopping;

import org.jwcarman.netwerx.EpochOutcome;

public class EpochCountStoppingAdvisor implements StoppingAdvisor {

// ------------------------------ FIELDS ------------------------------

    public static final int DEFAULT_MAX_EPOCHS = 100;
    private final int maxEpochs;

// --------------------------- CONSTRUCTORS ---------------------------

    public EpochCountStoppingAdvisor() {
        this(DEFAULT_MAX_EPOCHS);
    }

    public EpochCountStoppingAdvisor(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface StoppingAdvisor ---------------------

    @Override
    public boolean shouldStopAfter(EpochOutcome outcome) {
        return outcome.epoch() >= maxEpochs;
    }

}
