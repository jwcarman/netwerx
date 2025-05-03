package org.jwcarman.netwerx.stopping;

import org.jwcarman.netwerx.EpochOutcome;

public interface StoppingAdvisor {

// -------------------------- OTHER METHODS --------------------------

    boolean shouldStopAfter(EpochOutcome outcome);

}
