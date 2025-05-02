package org.jwcarman.netwerx;

import org.jwcarman.netwerx.activation.Activation;

public interface LayerConfig {
    LayerConfig activation(Activation activation);

    LayerConfig units(int units);
}
