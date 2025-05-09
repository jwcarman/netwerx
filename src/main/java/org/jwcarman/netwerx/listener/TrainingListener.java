package org.jwcarman.netwerx.listener;

import org.jwcarman.netwerx.network.EpochOutcome;

@FunctionalInterface
public interface TrainingListener {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Called at the end of each epoch during training.
     *
     * @param outcome the outcome of the epoch
     */
    void onEpoch(EpochOutcome outcome);

}
