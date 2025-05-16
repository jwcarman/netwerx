package org.jwcarman.netwerx.normalization;

import org.jwcarman.netwerx.util.stats.Stats;

import java.util.stream.DoubleStream;

public class L2NormalizationFunction implements NormalizationFunction {

// ------------------------------ FIELDS ------------------------------

    private final double norm;

// -------------------------- STATIC METHODS --------------------------

    public static L2NormalizationFunction fit(DoubleStream values) {
        var stats = Stats.of(values);
        return new L2NormalizationFunction(stats.l2());
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private L2NormalizationFunction(double norm) {
        this.norm = norm;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Normalizer ---------------------

    @Override
    public double normalize(double value) {
        if (norm == 0.0) {
            return 0.0;
        } else {
            return value / norm;
        }
    }

}
