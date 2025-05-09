package org.jwcarman.netwerx.parameter;

public class ParameterInitializers {

// -------------------------- STATIC METHODS --------------------------

    public static ParameterInitializer zeros() {
        return (_, _, _) -> 0.0;
    }

    public static ParameterInitializer constant(double value) {
        return (_, _, _) -> value;
    }

    public static ParameterInitializer xavierUniform() {
        return (rng, fanIn, fanOut) -> {
            double limit = Math.sqrt(6.0 / (fanIn + fanOut));
            return (rng.nextDouble() * 2 - 1) * limit;
        };
    }

    public static ParameterInitializer heUniform() {
        return (random, fanIn, _) -> {
            double limit = Math.sqrt(6.0 / fanIn);
            return (random.nextDouble() * 2 - 1) * limit;
        };
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private ParameterInitializers() {
        // Prevent instantiation
    }

}
