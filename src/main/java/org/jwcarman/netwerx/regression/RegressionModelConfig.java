package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.loss.Loss;

public interface RegressionModelConfig {
    RegressionModelConfig loss(Loss loss);
}
