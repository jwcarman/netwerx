package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.observer.TrainingObserver;

public interface RegressionModelTrainer<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    default RegressionModel<M> train(M inputs, double[] labels) {
        return train(inputs, labels, TrainingObserver.noop());
    }

    /**
     * Train a regression model using the provided inputs and labels.
     *
     * @param inputs   the feature matrix (shape: features x samples)
     * @param labels   the target values (shape: samples)
     * @param observer an observer to monitor the training process, which can be used to track progress, log information, or handle interruptions.
     * @return the trained regression model
     */
    RegressionModel<M> train(M inputs, double[] labels, TrainingObserver observer);

}
