package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.loss.Loss;

public interface BinaryClassifierConfig {
    BinaryClassifierConfig loss(Loss loss);
}
