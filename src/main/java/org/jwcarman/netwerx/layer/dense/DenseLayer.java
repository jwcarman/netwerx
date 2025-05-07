package org.jwcarman.netwerx.layer.dense;

import org.jwcarman.netwerx.activation.ActivationFunction;
import org.jwcarman.netwerx.layer.Layer;
import org.jwcarman.netwerx.matrix.Matrix;

public class DenseLayer<M extends Matrix<M>> implements Layer<M> {

// ------------------------------ FIELDS ------------------------------

    private final M weights;
    private final M biases;
    private final ActivationFunction activationFunction;

// --------------------------- CONSTRUCTORS ---------------------------

    public DenseLayer(M weights, M biases, ActivationFunction activationFunction) {
        this.weights = weights;
        this.biases = biases;
        this.activationFunction = activationFunction;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Layer ---------------------

    @Override
    public M apply(M input) {
        var z = weights.multiply(input).addColumnVector(biases);
        return activationFunction.apply(z);
    }

    @Override
    public int inputSize() {
        return weights.columnCount();
    }

    @Override
    public int outputSize() {
        return weights.rowCount();
    }
}
