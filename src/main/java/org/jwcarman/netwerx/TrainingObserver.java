package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;

public interface TrainingObserver {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Called at the end of each epoch during training.
     *
     * @param epoch the current epoch number
     * @param loss  the loss value for the epoch
     * @param a     the output of the model for the epoch
     * @param y     the target values for the epoch
     * @return true if training should continue, false to stop training
     */
    boolean onEpoch(int epoch, double loss, SimpleMatrix a, SimpleMatrix y);

}
