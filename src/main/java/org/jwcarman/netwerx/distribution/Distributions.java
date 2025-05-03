package org.jwcarman.netwerx.distribution;

import java.util.Random;

public class Distributions {

    private Distributions() {
        // Prevent instantiation
    }

    public static double symmetricUniform(Random rng, double scale) {
        return (rng.nextDouble() * 2 - 1) * scale;
    }

    public static double xavierUniform(Random rng, int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        return symmetricUniform(rng, limit);
    }

    public static double heUniform(Random rng, int fanIn) {
        double limit = Math.sqrt(6.0 / fanIn);
        return symmetricUniform(rng, limit);
    }
}
