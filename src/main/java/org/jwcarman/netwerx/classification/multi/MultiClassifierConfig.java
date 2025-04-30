package org.jwcarman.netwerx.classification.multi;

import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.optimization.Optimizer;

import java.util.Random;

public interface MultiClassifierConfig {
    MultiClassifierConfig biasOptimizer(Optimizer biasOptimizer);

    MultiClassifierConfig loss(Loss loss);

    MultiClassifierConfig optimizer(Optimizer optimizer);

    MultiClassifierConfig outputClasses(int outputClasses);

    MultiClassifierConfig random(Random random);

    MultiClassifierConfig weightOptimizer(Optimizer weightOptimizer);
}
