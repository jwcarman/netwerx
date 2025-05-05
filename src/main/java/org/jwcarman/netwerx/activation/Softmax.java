package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

public class Softmax implements ActivationFunction {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Activation ---------------------

    @Override
    public <M extends Matrix<M>> M apply(M logits) {
        return logits.columnSoftmax();
    }

    @Override
    public <M extends Matrix<M>> M derivative(M input) {
        return input.map((_, _, _) -> 1.0);
    }

}
