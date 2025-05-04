package org.jwcarman.netwerx.observer;

import org.slf4j.Logger;

public class TrainingObservers {
    private TrainingObservers() {
        // Prevent instantiation
    }

    /**
     * Returns a no-operation observer that does nothing when an epoch is completed.
     *
     * @return a noop TrainingObserver
     */
    public static TrainingObserver noop() {
        return _ -> {
            // No operation
        };
    }

    public static TrainingObserver logging(Logger logger, int step) {
        return outcome -> {
            if (outcome.epoch() % step == 0) {
                logger.info("Epoch {}: Training Loss = {}, Validation Loss = {}, Regularization Penalty = {}",
                        outcome.epoch(),
                        outcome.trainingLoss(),
                        outcome.validationLoss(),
                        outcome.regularizationPenalty());
            }
        };
    }
}
