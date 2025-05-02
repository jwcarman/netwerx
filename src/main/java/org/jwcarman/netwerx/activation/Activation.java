package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.Random;

public interface Activation {
    <M extends Matrix<M>> M apply(M input);

    <M extends Matrix<M>> M derivative(M input);

    default double generateInitialWeight(Random rand, int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        return (rand.nextDouble() * 2 - 1) * limit;
    }

    default double generateInitialBias() {
        return 0.0;
    }
}
