package org.jwcarman.netwerx.normalization;

import java.util.stream.DoubleStream;

public class MinMaxNormalizationFunction implements NormalizationFunction {
    public static final double DEFAULT_VALUE = 0.5;

// ------------------------------ FIELDS ------------------------------

    private final double min;
    private final double max;
    private final double range;

// -------------------------- STATIC METHODS --------------------------

    public static MinMaxNormalizationFunction fit(DoubleStream values) {
        var stats = values.summaryStatistics();
        return new MinMaxNormalizationFunction(stats.getMin(), stats.getMax());
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private MinMaxNormalizationFunction(double min, double max) {
        this.min = min;
        this.max = max;
        this.range = max - min;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Normalizer ---------------------

    @Override
    public double normalize(double value) {
        if(range == 0) {
            return DEFAULT_VALUE;
        }
        if (value < min) {
            return 0.0;
        } else if (value > max) {
            return 1.0;
        } else {
            return (value - min) / range;
        }
    }

}
