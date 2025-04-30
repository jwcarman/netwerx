package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.LayerConfig;
import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

class DefaultLayerConfig implements LayerConfig {

// ------------------------------ FIELDS ------------------------------

    private final int inputSize;
    private int units = 10;
    private Optimizer weightOptimizer = Optimizers.sgd();
    private Optimizer biasOptimizer = Optimizers.sgd();
    private Activation activation = Activations.relu();
    private Random random = Randoms.defaultRandom();

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultLayerConfig(int inputSize) {
        this.inputSize = inputSize;
    }

// --------------------- GETTER / SETTER METHODS ---------------------

    public Activation getActivation() {
        return activation;
    }

    public Optimizer getBiasOptimizer() {
        return biasOptimizer;
    }

    public int getInputSize() {
        return inputSize;
    }

    public Random getRandom() {
        return random;
    }

    public int getUnits() {
        return units;
    }

    public Optimizer getWeightOptimizer() {
        return weightOptimizer;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public DefaultLayerConfig activation(Activation activation) {
        this.activation = activation;
        return this;
    }

    @Override
    public DefaultLayerConfig biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    @Override
    public DefaultLayerConfig optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    @Override
    public DefaultLayerConfig random(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DefaultLayerConfig units(int units) {
        this.units = units;
        return this;
    }

    @Override
    public DefaultLayerConfig weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
