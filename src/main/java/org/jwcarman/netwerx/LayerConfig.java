package org.jwcarman.netwerx;

import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.optimization.Optimizer;

import java.util.Random;

public interface LayerConfig {
    LayerConfig activation(Activation activation);

    LayerConfig biasOptimizer(Optimizer biasOptimizer);

    LayerConfig optimizer(Optimizer optimizer);

    LayerConfig random(Random random);

    LayerConfig units(int units);

    LayerConfig weightOptimizer(Optimizer weightOptimizer);
}
