package org.jwcarman.netwerx.normalization;

import org.jwcarman.netwerx.util.stats.Stats;

import java.util.stream.DoubleStream;

public class ZScoreNormalizationFunction implements NormalizationFunction {

// ------------------------------ FIELDS ------------------------------

    private final double mean;
    private final double stdDev;

// -------------------------- STATIC METHODS --------------------------

    public static ZScoreNormalizationFunction fit(DoubleStream values) {
        var stats = Stats.of(values);
        return new ZScoreNormalizationFunction(stats.mean(), stats.stddev());
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private ZScoreNormalizationFunction(double mean, double stdDev) {
        this.mean = mean;
        this.stdDev = stdDev;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Normalizer ---------------------

    @Override
    public double normalize(double value) {
        return (value - mean) / stdDev;
    }

}
