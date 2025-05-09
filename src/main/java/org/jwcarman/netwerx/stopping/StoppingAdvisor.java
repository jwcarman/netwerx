package org.jwcarman.netwerx.stopping;

public interface StoppingAdvisor {

// -------------------------- OTHER METHODS --------------------------

    boolean shouldStop(int epoch, double score);

}
