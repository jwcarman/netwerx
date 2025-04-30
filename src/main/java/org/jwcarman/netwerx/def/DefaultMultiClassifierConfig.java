package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.classification.multi.MultiClassifierConfig;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

class DefaultMultiClassifierConfig implements MultiClassifierConfig {

// ------------------------------ FIELDS ------------------------------

    private Loss loss = Losses.cce();
    private Random random = Randoms.defaultRandom();
    private Optimizer weightOptimizer = Optimizers.sgd();
    private Optimizer biasOptimizer = Optimizers.sgd();
    private int outputClasses = 3;

// --------------------- GETTER / SETTER METHODS ---------------------

    public Optimizer getBiasOptimizer() {
        return biasOptimizer;
    }

    public Loss getLoss() {
        return loss;
    }

    public int getOutputClasses() {
        return outputClasses;
    }

    public Random getRandom() {
        return random;
    }

    public Optimizer getWeightOptimizer() {
        return weightOptimizer;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public DefaultMultiClassifierConfig biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    @Override
    public DefaultMultiClassifierConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

    @Override
    public DefaultMultiClassifierConfig optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    @Override
    public DefaultMultiClassifierConfig outputClasses(int outputClasses) {
        this.outputClasses = outputClasses;
        return this;
    }

    @Override
    public DefaultMultiClassifierConfig random(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DefaultMultiClassifierConfig weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
