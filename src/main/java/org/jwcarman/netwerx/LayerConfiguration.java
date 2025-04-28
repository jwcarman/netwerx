package org.jwcarman.netwerx;

import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.util.Randoms;

import java.util.Random;

public class LayerConfiguration {

// ------------------------------ FIELDS ------------------------------

    private final int inputSize;
    private int units = 10;
    private Optimizer weightOptimizer = Optimizers.sgd();
    private Optimizer biasOptimizer = Optimizers.sgd();
    private Activation activation = Activations.relu();
    private Random random = Randoms.defaultRandom();

// --------------------------- CONSTRUCTORS ---------------------------

    LayerConfiguration(int inputSize) {
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

    public LayerConfiguration activation(Activation activation) {
        this.activation = activation;
        return this;
    }

    public LayerConfiguration biasOptimizer(Optimizer biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    public LayerConfiguration optimizer(Optimizer optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    public LayerConfiguration random(Random random) {
        this.random = random;
        return this;
    }

    public LayerConfiguration units(int units) {
        this.units = units;
        return this;
    }

    public LayerConfiguration weightOptimizer(Optimizer weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
