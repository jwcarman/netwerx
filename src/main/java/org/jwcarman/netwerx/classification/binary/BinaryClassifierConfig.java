package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.optimization.Optimizer;

import java.util.Random;

public interface BinaryClassifierConfig {
    BinaryClassifierConfig biasOptimizer(Optimizer biasOptimizer);

    BinaryClassifierConfig loss(Loss loss);

    BinaryClassifierConfig optimizer(Optimizer optimizer);

    BinaryClassifierConfig random(Random random);

    BinaryClassifierConfig weightOptimizer(Optimizer weightOptimizer);
}
