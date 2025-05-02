package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.loss.Loss;

public interface MultiClassifierConfig {
    MultiClassifierConfig loss(Loss loss);
    MultiClassifierConfig outputClasses(int outputClasses);
}
