package org.jwcarman.netwerx;

import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

public class LayerConfig {

// ------------------------------ FIELDS ------------------------------

    private final int inputSize;
    private int units = 10;
    private Optimizer weightOptimizer = Optimizers.sgd();
    private Optimizer biasOptimizer = Optimizers.sgd();
    private Activation activation = Activations.relu();
    private Random random = Randoms.defaultRandom();

// --------------------------- CONSTRUCTORS ---------------------------

    LayerConfig(int inputSize) {
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

    public LayerConfig activation(Activation activation) {
        this.activation = activation;
        return this;
    }

    public LayerConfig biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    public LayerConfig optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    public LayerConfig random(Random random) {
        this.random = random;
        return this;
    }

    public LayerConfig units(int units) {
        this.units = units;
        return this;
    }

    public LayerConfig weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
