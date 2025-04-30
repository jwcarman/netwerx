package org.jwcarman.netwerx.regression;

import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.optimization.Optimizer;

import java.util.Random;

public interface RegressionModelConfig {
    RegressionModelConfig biasOptimizer(Optimizer biasOptimizer);

    RegressionModelConfig loss(Loss loss);

    RegressionModelConfig optimizer(Optimizer optimizer);

    RegressionModelConfig random(Random random);

    RegressionModelConfig weightOptimizer(Optimizer weightOptimizer);
}
