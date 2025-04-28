package org.jwcarman.netwerx.loss;

public interface LossObserver {
    void onEpoch(int epoch, double loss);
}
