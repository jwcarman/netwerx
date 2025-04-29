package org.jwcarman.netwerx.classification.binary;

import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

public class BinaryClassifierConfig {

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

    public BinaryClassifierConfig biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    public BinaryClassifierConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

    public BinaryClassifierConfig optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    public BinaryClassifierConfig random(Random random) {
        this.random = random;
        return this;
    }

    public BinaryClassifierConfig weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
