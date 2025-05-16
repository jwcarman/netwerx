package org.jwcarman.netwerx.normalization;

import org.jwcarman.netwerx.util.stats.Stats;

import java.util.stream.DoubleStream;

public class MaxAbsNormalizationFunction implements NormalizationFunction {

// ------------------------------ FIELDS ------------------------------

    private final double maxAbs;

// -------------------------- STATIC METHODS --------------------------

    public static MaxAbsNormalizationFunction fit(DoubleStream values) {
        var stats = Stats.of(values);
        return new MaxAbsNormalizationFunction(stats.maxAbs());
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private MaxAbsNormalizationFunction(double maxAbs) {
        this.maxAbs = maxAbs;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Normalizer ---------------------

    @Override
    public double normalize(double value) {
        if (maxAbs == 0.0) {
            return 0.0;
        } else {
            return value / maxAbs;
        }
    }

}
