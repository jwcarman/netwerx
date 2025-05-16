package org.jwcarman.netwerx.normalization;

import org.jwcarman.netwerx.util.stats.Stats;

import java.util.stream.DoubleStream;

public class LogNormalizationFunction implements NormalizationFunction {

// ------------------------------ FIELDS ------------------------------

    private final double minValue;

// -------------------------- STATIC METHODS --------------------------

    public static LogNormalizationFunction fit(DoubleStream values) {
        var stats = Stats.of(values);
        return new LogNormalizationFunction(stats.min());
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private LogNormalizationFunction(double minValue) {
        this.minValue = minValue;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Normalizer ---------------------

    @Override
    public double normalize(double value) {
        if (value <= 0.0) {
            return 0.0;
        } else {
            return Math.log1p(value - minValue);
        }
    }

}
