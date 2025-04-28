package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;

public class Softmax implements Activation {

    @Override
    public SimpleMatrix apply(SimpleMatrix input) {
        double max = input.elementMax();
        SimpleMatrix expInput = input.elementExp().divide(input.elementExp().elementSum());
        return expInput;
    }

    @Override
    public SimpleMatrix derivative(SimpleMatrix input) {
        // Softmax derivative is complex and typically not used directly.
        // It is often computed in the context of backpropagation.
        throw new UnsupportedOperationException("Softmax derivative is not implemented.");
    }
}
