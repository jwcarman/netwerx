package org.jwcarman.netwerx.score;

import org.jwcarman.netwerx.EpochOutcome;

@FunctionalInterface
public interface ScoringFunction {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Computes a score for the given epoch outcome.
     * Higher scores are considered better for the purposes of early stopping and model checkpointing.
     *
     * @param outcome the outcome of the current epoch
     * @return a score indicating the quality of this epoch (higher is better)
     */
    double score(EpochOutcome outcome);

}
