package org.jwcarman.netwerx;

import org.jwcarman.netwerx.matrix.Matrix;

public interface TrainingObserver {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Called at the end of each epoch during training.
     *
     * @param epoch the current epoch number
     * @param loss  the loss value for the epoch
     * @param yHat     the output of the model for the epoch
     * @param y     the target values for the epoch
     * @return true if training should continue, false to stop training
     */
    <M extends Matrix<M>> boolean onEpoch(int epoch, double loss, M yHat, M y);

}
