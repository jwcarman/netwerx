package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

public class Softmax implements ActivationFunction {

// ------------------------------ FIELDS ------------------------------

    public static final ActivationFunction INSTANCE = new Softmax();

// --------------------------- CONSTRUCTORS ---------------------------

    private Softmax() {
        // singleton
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ActivationFunction ---------------------

    @Override
    public <M extends Matrix<M>> M apply(M logits) {
        return logits.columnSoftmax();
    }

    @Override
    public <M extends Matrix<M>> M derivative(M input) {
        return input.map((_, _, _) -> 1.0);
    }

}
