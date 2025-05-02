package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.LayerConfig;
import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.activation.Activations;

class DefaultLayerConfig implements LayerConfig {

// ------------------------------ FIELDS ------------------------------

    private final int inputSize;
    private int units = 10;
    private Activation activation = Activations.relu();

// --------------------------- CONSTRUCTORS ---------------------------

    public DefaultLayerConfig(int inputSize) {
        this.inputSize = inputSize;
    }

// --------------------- GETTER / SETTER METHODS ---------------------

    public Activation getActivation() {
        return activation;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getUnits() {
        return units;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface LayerConfig ---------------------

    @Override
    public DefaultLayerConfig activation(Activation activation) {
        this.activation = activation;
        return this;
    }

    @Override
    public DefaultLayerConfig units(int units) {
        this.units = units;
        return this;
    }

}
