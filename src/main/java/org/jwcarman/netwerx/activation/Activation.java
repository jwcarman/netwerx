package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public interface Activation {
    SimpleMatrix apply(SimpleMatrix input);

    SimpleMatrix derivative(SimpleMatrix input);

    default double generateInitialWeight(Random rand, int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        return (rand.nextDouble() * 2 - 1) * limit;
    }

    default double generateInitialBias() {
        return 0.0;
    }
}
