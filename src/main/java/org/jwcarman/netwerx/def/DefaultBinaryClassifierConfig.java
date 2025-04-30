package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.classification.binary.BinaryClassifierConfig;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

class DefaultBinaryClassifierConfig implements BinaryClassifierConfig {

// ------------------------------ FIELDS ------------------------------

    private Loss loss = Losses.bce();
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
    public DefaultBinaryClassifierConfig biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    @Override
    public DefaultBinaryClassifierConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

    @Override
    public DefaultBinaryClassifierConfig optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    @Override
    public DefaultBinaryClassifierConfig random(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DefaultBinaryClassifierConfig weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
