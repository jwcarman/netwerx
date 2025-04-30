package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.regression.RegressionModelConfig;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

class DefaultRegressionModelConfig implements RegressionModelConfig {

// ------------------------------ FIELDS ------------------------------

    private Loss loss = Losses.mse();
    private Random random = Randoms.defaultRandom();
    private Optimizer weightOptimizer = Optimizers.sgd();
    private Optimizer biasOptimizer = Optimizers.sgd();

// --------------------- GETTER / SETTER METHODS ---------------------

    public Optimizer getBiasOptimizer() {
        return biasOptimizer;
    }

    public Loss getLoss() {
        return loss;
    }

    public Random getRandom() {
        return random;
    }

    public Optimizer getWeightOptimizer() {
        return weightOptimizer;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public DefaultRegressionModelConfig biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    @Override
    public DefaultRegressionModelConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

    @Override
    public DefaultRegressionModelConfig optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    @Override
    public DefaultRegressionModelConfig random(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DefaultRegressionModelConfig weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
