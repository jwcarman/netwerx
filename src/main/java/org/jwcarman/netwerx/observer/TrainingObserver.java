package org.jwcarman.netwerx.observer;

import org.jwcarman.netwerx.EpochOutcome;

@FunctionalInterface
public interface TrainingObserver {

// -------------------------- OTHER METHODS --------------------------

    static TrainingObserver noop() {
        return outcome -> {
            // No operation
        };
    }

    /**
     * Called at the end of each epoch during training.
     *
     * @param outcome the outcome of the epoch
     */
    void onEpoch(EpochOutcome outcome);

}
