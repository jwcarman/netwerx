package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

public class Softmax implements ActivationFunction {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Activation ---------------------

    @Override
    public <M extends Matrix<M>> M apply(M logits) {
        return logits.softmax();
    }

    @Override
    public <M extends Matrix<M>> M derivative(M input) {
        return input.map((row, col, value) -> 1.0);
    }

}
